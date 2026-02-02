import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="Olist 리뷰 분석 대시보드", layout="wide")

# 데이터 로드 함수 (캐싱 사용)
@st.cache_data
def load_data():
    # 데이터 경로 설정
    pq_path = r'c:\fcicb6\data\OLIST_V.2\DATA_PARQUET'
    local_path = r'c:\fcicb6\data\OLIST_V.2\DATA_REV.2'
    
    # 1. Parquet 경로 우선 탐색, 없으면 CSV 경로 혹은 상대 경로 사용
    if os.path.exists(pq_path):
        base_path = pq_path
        ext = '.parquet'
    elif os.path.exists(local_path):
        base_path = local_path
        ext = '.csv'
    else:
        # 클라우드 환경 대응
        base_path = os.path.join(os.path.dirname(__file__), 'DATA_PARQUET')
        ext = '.parquet'
        if not os.path.exists(base_path):
            base_path = os.path.join(os.path.dirname(__file__), 'DATA_REV.2')
            ext = '.csv'
    
    def read_df(name):
        full_path = os.path.join(base_path, f'{name}{ext}')
        return pd.read_parquet(full_path) if ext == '.parquet' else pd.read_csv(full_path)

    # 데이터 읽기
    orders = read_df('proc_olist_orders_dataset')
    items = read_df('proc_olist_order_items_dataset')
    reviews = read_df('proc_olist_order_reviews_dataset')
    payments = read_df('proc_olist_order_payments_dataset')
    customers = read_df('proc_olist_customers_dataset')
    
    # 시간 데이터 변환 (Parquet은 이미 타입이 보존될 수 있으나 안전을 위해 체크)
    date_cols = [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        if col in orders.columns and not pd.api.types.is_datetime64_any_dtype(orders[col]):
            orders[col] = pd.to_datetime(orders[col])
            
    # 1. 전처리: delay_days 계산 및 배송 기간 계산
    # 배송 완료된 주문만 대상으로 지연 일수 계산 (NaT 처리 포함)
    orders['delay_days'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
    orders['shipping_duration'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
    
    # 2. 전처리: 배송비 비중 (freight_ratio)
    items['freight_ratio'] = items['freight_value'] / items['price']
    order_items_summary = items.groupby('order_id').agg({
        'price': 'sum',
        'freight_value': 'sum',
        'freight_ratio': 'mean'
    }).reset_index()
    
    # 3. 데이터 병합
    df = orders.merge(reviews[['order_id', 'review_score']], on='order_id', how='inner')
    df = df.merge(order_items_summary, on='order_id', how='inner')
    df = df.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='inner')
    
    # 리뷰 그룹 생성
    df['review_group'] = df['review_score'].apply(lambda x: 'High (4-5)' if x >= 4 else 'Low (1-3)')
    
    # 4. RFM 계산
    reference_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).rename(columns={
        'order_purchase_timestamp': 'Recency',
        'order_id': 'Frequency',
        'price': 'Monetary'
    })
    
    # RFM 점수
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = rfm['Frequency'].rank(method='first').transform(lambda x: pd.qcut(x, 5, labels=[1, 2, 3, 4, 5]))
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    
    def segment_customer(row):
        score = int(row['R_Score']) + int(row['F_Score']) + int(row['M_Score'])
        if score >= 13: return 'VIP'
        elif score >= 9: return 'Loyal'
        elif score >= 5: return 'Regular'
        else: return 'At Risk'
        
    rfm['RFM_Segment'] = rfm.apply(segment_customer, axis=1)
    df = df.merge(rfm[['RFM_Segment']], on='customer_unique_id', how='left')
    
    return df, payments

# 데이터 로드
df, payments = load_data()

# 사이드바 - RFM 필터
st.sidebar.title("📊 필터 설정")
selected_segments = st.sidebar.multiselect(
    "RFM 세그먼트 선택",
    options=df['RFM_Segment'].unique(),
    default=df['RFM_Segment'].unique()
)

# 필터링된 데이터
df_filtered = df[df['RFM_Segment'].isin(selected_segments)]

# 메인 타이틀
st.title("🛍️ Olist 리뷰 점수 그룹별 비교 대시보드")
st.markdown("리뷰 점수에 따른 물류 성과와 결제 패턴의 차이를 분석합니다.")

# 지표 요약
col1, col2, col3, col4 = st.columns(4)
col1.metric("총 주문 건수", f"{len(df_filtered):,}")
col2.metric("평균 리뷰 점수", f"{df_filtered['review_score'].mean():.2f}")
col3.metric("평균 배송 지연 (일)", f"{df_filtered['delay_days'].mean():.2f}")
col4.metric("평균 배송비 비중", f"{df_filtered['freight_ratio'].mean()*100:.1f}%")

st.divider()

# 시각화 1 (물류 비교)
st.subheader("🚚 물류 성과 비교 (Low vs High Group)")

logistics_comp = df_filtered.groupby('review_group').agg({
    'shipping_duration': 'mean',
    'delay_days': 'mean',
    'freight_ratio': 'mean'
}).reset_index()

# 데이터 재구성 (Plotly Bar Chart용)
logistics_melted = logistics_comp.melt(id_vars='review_group', 
                                      value_vars=['shipping_duration', 'delay_days', 'freight_ratio'],
                                      var_name='Metric', value_name='Value')

# 한글 라벨링
metric_map = {
    'shipping_duration': '평균 배송 기간',
    'delay_days': '평균 지연 일수',
    'freight_ratio': '평균 배송비 비중'
}
logistics_melted['Metric_KR'] = logistics_melted['Metric'].map(metric_map)

fig1 = px.bar(logistics_melted, x='Metric_KR', y='Value', color='review_group',
             barmode='group', text_auto='.2f',
             title="리뷰 그룹별 주요 물류 지표 비교",
             labels={'Value': '값', 'Metric_KR': '지표', 'review_group': '리뷰 그룹'},
             color_discrete_map={'High (4-5)': '#00CC96', 'Low (1-3)': '#EF553B'})

st.plotly_chart(fig1, use_container_width=True)

with st.expander("💡 물류 인사이트"):
    st.info("""
    - **배송 지연**은 리뷰 점수에 결정적인 영향을 미칩니다. Low 그룹은 High 그룹에 비해 평균 지연 일수가 확연히 높습니다.
    - **배송비 비중**이 높을수록 고객의 기대치가 높아져, 작은 지연에도 더 민감하게 낮은 점수를 줄 가능성이 있습니다.
    """)

st.divider()

# 시각화 2 (결제 수단 비교)
st.subheader("💳 리뷰 그룹별 결제 수단 사용 비중")

# 각 그룹별 주문 목록 추출
low_orders = df_filtered[df_filtered['review_group'] == 'Low (1-3)']['order_id']
high_orders = df_filtered[df_filtered['review_group'] == 'High (4-5)']['order_id']

# 결제 수단 비중 계산
low_payments = payments[payments['order_id'].isin(low_orders)]['payment_type'].value_counts(normalize=True).reset_index()
low_payments['review_group'] = 'Low (1-3)'

high_payments = payments[payments['order_id'].isin(high_orders)]['payment_type'].value_counts(normalize=True).reset_index()
high_payments['review_group'] = 'High (4-5)'

payment_comp = pd.concat([low_payments, high_payments])
payment_comp.columns = ['payment_type', 'proportion', 'review_group']

# Sunburst Chart
fig2 = px.sunburst(payment_comp, path=['review_group', 'payment_type'], values='proportion',
                  color='payment_type',
                  title="리뷰 그룹별 결제 수단 분포",
                  labels={'proportion': '비중', 'payment_type': '결제 수단', 'review_group': '리뷰 그룹'})

st.plotly_chart(fig2, use_container_width=True)

# 결제 수단별 건수 집계 (추가 요구사항)
with st.expander("📝 결제 수단별 주문 건수 상세 보기"):
    payment_counts = payments[payments['order_id'].isin(df_filtered['order_id'])]
    payment_summary = pd.merge(
        payment_counts, 
        df_filtered[['order_id', 'review_group']], 
        on='order_id'
    ).groupby(['review_group', 'payment_type']).size().reset_index(name='주문 건수')
    st.table(payment_summary.pivot(index='payment_type', columns='review_group', values='주문 건수').fillna(0))

# 바우처 결제 강조를 위한 별도 분석
voucher_low = payment_comp[(payment_comp['review_group'] == 'Low (1-3)') & (payment_comp['payment_type'] == 'voucher')]['proportion']
voucher_high = payment_comp[(payment_comp['review_group'] == 'High (4-5)') & (payment_comp['payment_type'] == 'voucher')]['proportion']

v_low = voucher_low.values[0] if not voucher_low.empty else 0
v_high = voucher_high.values[0] if not voucher_high.empty else 0

st.subheader("📢 결제 수단 특이사항")
col_v1, col_v2 = st.columns(2)
col_v1.metric("Low 그룹 바우처 비중", f"{v_low*100:.1f}%")
col_v2.metric("High 그룹 바우처 비중", f"{v_high*100:.1f}%")

st.warning(f"분석 결과: **Low 점수 그룹**의 바우처 결제 비중이 **High 그룹** 대비 약 {v_low/v_high if v_high > 0 else 0:.1f}배 높게 나타납니다. 이는 환불 혹은 보상성 바우처 사용이 재구매 시 낮은 만족도로 이어졌을 가능성을 시사합니다.")

# VIP 등급 분석
if 'VIP' in selected_segments:
    st.divider()
    st.subheader("🌟 VIP 등급 저만족(Low) 원인 분석")
    vip_df = df_filtered[(df_filtered['RFM_Segment'] == 'VIP')]
    vip_low = vip_df[vip_df['review_group'] == 'Low (1-3)']
    
    if len(vip_low) > 0:
        st.write(f"현재 선택된 필터 내 VIP 등급 중 저만족(1-3점) 고객은 **{len(vip_low)}명**입니다.")
        avg_delay_vip_low = vip_low['delay_days'].mean()
        avg_delay_vip_high = vip_df[vip_df['review_group'] == 'High (4-5)']['delay_days'].mean()
        
        c1, c2 = st.columns(2)
        c1.write("**VIP 저만족 그룹 평균 지연:**")
        c1.error(f"{avg_delay_vip_low:.1f} 일")
        c2.write("**VIP 고만족 그룹 평균 지연:**")
        c2.success(f"{avg_delay_vip_high:.1f} 일")
        
        st.info("VIP 고객은 일반 고객보다 배송 지연에 더 민감할 수 있습니다. VIP 등급의 저만족 원인 중 상당수가 '배송 약속 미이행'에 있음을 보여줍니다.")
    else:
        st.write("선택된 필터 내 VIP 등급 중 저만족 고객이 없습니다. 우수한 충성도를 유지하고 있습니다.")
