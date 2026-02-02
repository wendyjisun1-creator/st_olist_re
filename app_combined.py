import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="Olist 통합 분석 대시보드", layout="wide")

# 데이터 로드 함수 (캐싱 사용)
@st.cache_data
def load_all_combined_data():
    # 데이터 폴더 후보군 (로컬 및 클라우드 환경)
    possible_paths = [
        r'c:\fcicb6\data\OLIST_V.2\DATA_PARQUET', # 로컬 파케
        os.path.join(os.path.dirname(__file__), 'DATA_PARQUET'), # 클라우드 파케 폴더
        os.path.join(os.path.dirname(__file__), 'data'), # 'data' 폴더 내
        os.path.dirname(__file__), # 루트 폴더
        r'c:\fcicb6\data\OLIST_V.2\DATA_REV.2', # 로컬 원본 (CSV)
    ]
    
    base_path = None
    ext = None
    
    # 1. Parquet 파일 확인
    for p in possible_paths:
        if os.path.exists(p) and os.path.exists(os.path.join(p, 'proc_olist_orders_dataset.parquet')):
            base_path = p
            ext = '.parquet'
            break
            
    # 2. CSV 확인 (Parquet 없으면)
    if not base_path:
        for p in possible_paths:
            if os.path.exists(p) and os.path.exists(os.path.join(p, 'proc_olist_orders_dataset.csv')):
                base_path = p
                ext = '.csv'
                break
                
    if not base_path:
        st.error("데이터 파일을 찾을 수 없습니다. 파케(Parquet) 또는 CSV 파일이 올바른 위치에 있는지 확인해주세요.")
        st.stop()
    
    def read_df(name):
        full_path = os.path.join(base_path, f'{name}{ext}')
        if ext == '.parquet':
            return pd.read_parquet(full_path)
        else:
            return pd.read_csv(full_path)

    # 데이터 로딩
    orders = read_df('proc_olist_orders_dataset')
    items = read_df('proc_olist_order_items_dataset')
    reviews = read_df('proc_olist_order_reviews_dataset')
    payments = read_df('proc_olist_order_payments_dataset')
    customers = read_df('proc_olist_customers_dataset')
    products = read_df('proc_olist_products_dataset')
    
    # 시간 데이터 변환
    date_cols = [
        'order_purchase_timestamp', 'order_approved_at', 
        'order_delivered_carrier_date', 'order_delivered_customer_date', 
        'order_estimated_delivery_date'
    ]
    for col in date_cols:
        if col in orders.columns and not pd.api.types.is_datetime64_any_dtype(orders[col]):
            orders[col] = pd.to_datetime(orders[col])
            
    # 1. 기본 전처리: 지연 일수 및 배송 기간
    orders['delay_days'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
    orders['shipping_duration'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
    
    # 2. 아이템 정보 요약 (배송비 비중 등)
    items['freight_ratio'] = items['freight_value'] / items['price']
    
    # 3. 데이터 병합 (분석용 메인 데이터셋)
    # orders 기준으로 items, reviews, customers, products 병합
    df = orders.merge(items, on='order_id', how='inner')
    df = df.merge(reviews[['order_id', 'review_score']], on='order_id', how='left')
    df = df.merge(customers[['customer_id', 'customer_unique_id']], on='customer_id', how='inner')
    df = df.merge(products[['product_id', 'product_category_name_english']], on='product_id', how='left')
    
    # 리뷰 그룹 생성
    df['review_group'] = df['review_score'].apply(lambda x: 'High (4-5)' if x >= 4 else ('Low (1-3)' if x >= 1 else 'None'))
    
    # 4. RFM 계산 (customer_unique_id 기준)
    ref_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (ref_date - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).rename(columns={'order_purchase_timestamp': 'Recency', 'order_id': 'Frequency', 'price': 'Monetary'})
    
    # RFM 점수 (1-5등급)
    rfm['R'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F'] = rfm['Frequency'].rank(method='first').transform(lambda x: pd.qcut(x, 5, labels=[1,2,3,4,5]))
    rfm['M'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5])
    
    def get_segment(row):
        score = int(row['R']) + int(row['F']) + int(row['M'])
        if score >= 13: return 'VIP'
        elif score >= 9: return 'Loyal'
        elif score >= 5: return 'Regular'
        else: return 'At Risk'
        
    rfm['RFM_Segment'] = rfm.apply(get_segment, axis=1)
    df = df.merge(rfm[['RFM_Segment']], on='customer_unique_id', how='left')
    
    return df, payments

# 데이터 로드
df_all, payments_raw = load_all_combined_data()

# --- 사이드바 필터 ---
st.sidebar.title("🔍 통합 필터 설정")

# 1. 날짜 필터
min_date = df_all['order_purchase_timestamp'].min().to_pydatetime()
max_date = df_all['order_purchase_timestamp'].max().to_pydatetime()
date_range = st.sidebar.date_input("조문 기간 선택", [min_date, max_date], min_value=min_date, max_value=max_date)

# 2. RFM 세그먼트 필터
segments = sorted(df_all['RFM_Segment'].unique())
selected_segments = st.sidebar.multiselect("RFM 세그먼트 선택", options=segments, default=segments)

# 필터 적용
if len(date_range) == 2:
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    mask = (df_all['order_purchase_timestamp'] >= start_date) & \
           (df_all['order_purchase_timestamp'] <= end_date) & \
           (df_all['RFM_Segment'].isin(selected_segments))
    df_filtered = df_all.loc[mask]
else:
    df_filtered = df_all[df_all['RFM_Segment'].isin(selected_segments)]

# 필터링된 데이터 기반 결제 데이터 필터링
payments_filtered = payments_raw[payments_raw['order_id'].isin(df_filtered['order_id'])]

# --- 메인 대시보드 구조 ---
st.title("🛒 Olist 데이터 통합 분석 대시보드")
st.markdown("매출 트렌드, 카테고리 성과, 리뷰 및 물류 지표를 한눈에 확인하세요.")

# KPI 요약
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("총 매출액", f"R$ {df_filtered['price'].sum():,.0f}")
kpi2.metric("총 주문 건수", f"{df_filtered['order_id'].nunique():,}건")
kpi3.metric("평균 리뷰 점수", f"{df_filtered['review_score'].mean():.2f}점")
kpi4.metric("분석 대상 고객수", f"{df_filtered['customer_unique_id'].nunique():,}명")

st.divider()

# 탭 구성
tab1, tab2 = st.tabs(["📊 매출 및 판매 트렌드", "🚚 리뷰 및 물류 성과"])

# --- TAB 1: 매출 및 판매 트렌드 ---
with tab1:
    st.header("📈 매출 및 판매 트렌드 분석")
    
    # 시각화 1: 월별 추이 (이중 축)
    trend_df = df_filtered.copy()
    trend_df['month'] = trend_df['order_purchase_timestamp'].dt.to_period('M').astype(str)
    trend_data = trend_df.groupby('month').agg({'price': 'sum', 'order_id': 'nunique'}).reset_index()

    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trend.add_trace(go.Scatter(x=trend_data['month'], y=trend_data['price'], name="매출액 (R$)", mode='lines+markers', line=dict(color='#636EFA')), secondary_y=False)
    fig_trend.add_trace(go.Scatter(x=trend_data['month'], y=trend_data['order_id'], name="판매량 (건)", mode='lines+markers', line=dict(color='#EF553B', dash='dot')), secondary_y=True)
    
    fig_trend.update_layout(title="월별 매출액 및 판매량 추이", hovermode="x unified")
    fig_trend.update_yaxes(title_text="매출액 (R$)", secondary_y=False)
    fig_trend.update_yaxes(title_text="판매량 (건)", secondary_y=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    # 시각화 2: 카테고리 Treemap
    st.subheader("🌳 카테고리별 매출 및 만족도 (상위 10)")
    cat_data = df_filtered.groupby('product_category_name_english').agg({'price': 'sum', 'review_score': 'mean'}).reset_index()
    cat_top10 = cat_data.nlargest(10, 'price')

    fig_tree = px.treemap(cat_top10, path=['product_category_name_english'], values='price',
                         color='review_score', color_continuous_scale='RdYlGn',
                         title="카테고리별 매출 규모와 평균 평점 (녹색: 높음, 적색: 낮음)")
    st.plotly_chart(fig_tree, use_container_width=True)

    # 시각화 3: 리뷰 개수 vs 판매량 산점도
    st.subheader("🔍 리뷰 영향력 분석 (리뷰 개수와 판매량 상관관계)")
    prod_analysis = df_filtered.groupby('product_id').agg({'review_score': 'count', 'order_id': 'nunique'}).reset_index()
    prod_analysis.columns = ['product_id', 'review_count', 'sales_volume']
    
    # 아웃라이어 정제 (가시성)
    q_limit = prod_analysis['sales_volume'].quantile(0.99)
    fig_scatter = px.scatter(prod_analysis[prod_analysis['sales_volume'] <= q_limit], 
                            x='review_count', y='sales_volume', trendline="ols",
                            opacity=0.5, title="상품별 리뷰 개수와 판매량 상관관계 (추세선 포함)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 2: 리뷰 및 물류 성과 ---
with tab2:
    st.header("🚚 리뷰 점수 그룹별 물류 및 결제 분석")
    
    # 1. 물류 지표 비교 (Grouped Bar Chart)
    st.subheader("📦 리뷰 그룹별 주요 물류 지표")
    log_df = df_filtered[df_filtered['review_group'] != 'None'].groupby('review_group').agg({
        'shipping_duration': 'mean',
        'delay_days': 'mean',
        'freight_ratio': 'mean'
    }).reset_index()
    
    log_melted = log_df.melt(id_vars='review_group', 
                            value_vars=['shipping_duration', 'delay_days', 'freight_ratio'],
                            var_name='Metric', value_name='Value')
    
    metric_naming = {'shipping_duration': '평균 배송일', 'delay_days': '평균 지연일', 'freight_ratio': '배송비 비중'}
    log_melted['Metric_KR'] = log_melted['Metric'].map(metric_naming)
    
    fig_log = px.bar(log_melted, x='Metric_KR', y='Value', color='review_group', barmode='group',
                    text_auto='.2f', title="리뷰 그룹별 물류 성과 비교",
                    color_discrete_map={'High (4-5)': '#00CC96', 'Low (1-3)': '#EF553B'})
    st.plotly_chart(fig_log, use_container_width=True)

    # 2. 결제 수단 비중 (Sunburst)
    st.subheader("💳 리뷰 그룹별 결제 수단분포")
    # 그룹별 결제 데이터 구성
    pay_type_comp = []
    for grp in ['High (4-5)', 'Low (1-3)']:
        grp_orders = df_filtered[df_filtered['review_group'] == grp]['order_id']
        grp_pay = payments_raw[payments_raw['order_id'].isin(grp_orders)]['payment_type'].value_counts(normalize=True).reset_index()
        grp_pay['review_group'] = grp
        pay_type_comp.append(grp_pay)
    
    pay_final = pd.concat(pay_type_comp)
    pay_final.columns = ['payment_type', 'proportion', 'review_group']
    
    fig_sun = px.sunburst(pay_final, path=['review_group', 'payment_type'], values='proportion',
                         color='payment_type', title="리뷰 그룹별 결제 수단 비중")
    st.plotly_chart(fig_sun, use_container_width=True)

    # 바우처 강조
    v_low = pay_final[(pay_final['review_group'] == 'Low (1-3)') & (pay_final['payment_type'] == 'voucher')]['proportion'].values
    v_high = pay_final[(pay_final['review_group'] == 'High (4-5)') & (pay_final['payment_type'] == 'voucher')]['proportion'].values
    
    v_l = v_low[0] if len(v_low) > 0 else 0
    v_h = v_high[0] if len(v_high) > 0 else 0
    
    st.info(f"💡 Low 그룹의 바우처 결제 비중은 **{v_l*100:.1f}%**로, High 그룹(**{v_h*100:.1f}%**)보다 높게 나타납니다. (보상성 결제 가능성)")

    # 3. VIP 심층 분석
    if 'VIP' in selected_segments:
        st.divider()
        st.subheader("🌟 VIP 등급 집중 분석")
        vip_data = df_filtered[df_filtered['RFM_Segment'] == 'VIP']
        vip_low = vip_data[vip_data['review_group'] == 'Low (1-3)']
        
        if len(vip_low) > 0:
            st.warning(f"분석 기간 내 VIP 고객 중 **{len(vip_low)}건**의 낮은 만족도(1-3점)가 발생했습니다.")
            v_c1, v_c2 = st.columns(2)
            v_c1.metric("VIP Low 그룹 평균 지연", f"{vip_low['delay_days'].mean():.1f}일", delta_color="inverse")
            v_c2.metric("VIP High 그룹 평균 지연", f"{vip_data[vip_data['review_group'] == 'High (4-5)']['delay_days'].mean():.1f}일")
            st.write("VIP 고객의 이탈을 막기 위해 지연된 배송에 대한 타겟 케어가 필요합니다.")
        else:
            st.success("분석 기간 내 모든 VIP 고객이 높은 만족도를 유지하고 있습니다!")

# --- 하단 인사이트 및 결론 ---
st.divider()
st.subheader("💡 데이터 기반 종합 분석 결과")
ins1, ins2, ins3 = st.columns(3)

with ins1:
    st.markdown("### 📈 성장 동력 (Growth)")
    st.write("- 매출과 판매량은 리뷰 개수와 강력한 양의 상관관계를 가집니다.")
    st.write("- **신규 제품**의 빠른 시장 안착을 위해 초기 리뷰 확보 캠페인이 필수적입니다.")

with ins2:
    st.markdown("### 🚚 운영 최적화 (Logistics)")
    st.write("- 리뷰 점수를 가르는 결정적 요인은 **'배송 지연'**입니다.")
    st.write("- 특히 Low 그룹의 지연 일수가 월등히 높은 점을 고려할 때, 물류 효율 개선이 만족도 향상의 직결타입니다.")

with ins3:
    st.markdown("### 🎯 고객 유지 (Retention)")
    st.write("- VIP 고객의 만족도 저하는 배송 지연 시 더 두드러집니다.")
    st.write("- 바우처 사용 비중이 높은 고객군에 대한 재구매 유도 및 서비스 사후 관리가 필요합니다.")

st.success("🎯 **통합 전략:** 리뷰가 많은 카테고리의 물류 품질을 우선적으로 관리하여, '리뷰 증대 → 매출 상승 → 우수한 고객 경험'의 선순환 구조를 구축해야 합니다.")
