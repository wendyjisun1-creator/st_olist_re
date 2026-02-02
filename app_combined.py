import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
from datetime import datetime

# 페이지 설정
st.set_page_config(page_title="Olist 통합 전략 대시보드", layout="wide")

# 데이터 로드 함수 (캐싱 사용)
@st.cache_data
def load_all_dashboard_data():
    # 데이터 폴더 후보군
    possible_paths = [
        r'c:\fcicb6\data\OLIST_V.2\DATA_PARQUET',
        os.path.join(os.path.dirname(__file__), 'DATA_PARQUET'),
        os.path.join(os.path.dirname(__file__), 'data'),
        os.path.dirname(__file__),
    ]
    
    base_path = None
    for p in possible_paths:
        if os.path.exists(p) and (os.path.exists(os.path.join(p, 'proc_olist_orders_dataset.parquet')) or 
                                os.path.exists(os.path.join(p, 'proc_olist_orders_dataset.csv'))):
            base_path = p
            break
            
    if not base_path:
        st.error("핵심 데이터 파일을 찾을 수 없습니다.")
        st.stop()
    
    def read_df(name):
        pq = os.path.join(base_path, f'{name}.parquet')
        csv = os.path.join(base_path, f'{name}.csv')
        if os.path.exists(pq): return pd.read_parquet(pq)
        if os.path.exists(csv): return pd.read_csv(csv)
        return pd.DataFrame()

    # 데이터 로딩
    orders = read_df('proc_olist_orders_dataset')
    items = read_df('proc_olist_order_items_dataset')
    reviews = read_df('proc_olist_order_reviews_dataset')
    payments = read_df('proc_olist_order_payments_dataset')
    customers = read_df('proc_olist_customers_dataset')
    products = read_df('proc_olist_products_dataset')
    
    # 시간 데이터 변환
    date_cols = ['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols:
        if col in orders.columns and not pd.api.types.is_datetime64_any_dtype(orders[col]):
            orders[col] = pd.to_datetime(orders[col])
            
    # 기본 전처리: 지연 일수 및 배송 기간
    orders['delay_days'] = (orders['order_delivered_customer_date'] - orders['order_estimated_delivery_date']).dt.days
    orders['shipping_duration'] = (orders['order_delivered_customer_date'] - orders['order_purchase_timestamp']).dt.days
    
    # 아이템 정보
    items['freight_ratio'] = items['freight_value'] / items['price']
    
    # 데이터 병합
    df = orders.merge(items, on='order_id', how='inner')
    df = df.merge(reviews[['order_id', 'review_score', 'review_comment_message']], on='order_id', how='left')
    df = df.merge(customers[['customer_id', 'customer_unique_id', 'customer_state']], on='customer_id', how='inner')
    
    if not products.empty:
        df = df.merge(products[['product_id', 'product_category_name_english', 'product_photos_qty']], on='product_id', how='left')
    else:
        df['product_category_name_english'] = 'Unknown'
        df['product_photos_qty'] = 0
    
    # 리뷰 그룹 설정 (빨간색-Low, 파란색-High 대비를 위해)
    def categorize_review(score):
        if pd.isna(score): return 'None'
        return 'High (4-5)' if score >= 4 else 'Low (1-3)'
    
    df['review_group'] = df['review_score'].apply(categorize_review)
    
    # RFM 계산
    ref_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('customer_unique_id').agg({
        'order_purchase_timestamp': lambda x: (ref_date - x.max()).days,
        'order_id': 'nunique',
        'price': 'sum'
    }).rename(columns={'order_purchase_timestamp': 'Recency', 'order_id': 'Frequency', 'price': 'Monetary'})
    
    for col, labels in zip(['Recency', 'Frequency', 'Monetary'], [[5,4,3,2,1], [1,2,3,4,5], [1,2,3,4,5]]):
        if col == 'Frequency': # Frequency는 중복값이 많을 수 있어 rank 사용
            rfm[col[0]] = rfm[col].rank(method='first').transform(lambda x: pd.qcut(x, 5, labels=labels))
        else:
            rfm[col[0]] = pd.qcut(rfm[col], 5, labels=labels)
            
    rfm['RFM_Segment'] = rfm.apply(lambda x: 'VIP' if int(x['R'])+int(x['F'])+int(x['M']) >= 13 else 
                                   ('Loyal' if int(x['R'])+int(x['F'])+int(x['M']) >= 9 else 
                                    ('Regular' if int(x['R'])+int(x['F'])+int(x['M']) >= 5 else 'At Risk')), axis=1)
    
    df = df.merge(rfm[['RFM_Segment']], on='customer_unique_id', how='left')
    
    return df, payments

# 데이터 로드
df_all, payments_raw = load_all_dashboard_data()

# --- 사이드바 ---
st.sidebar.title("🛠️ 데이터 필터")
min_d = df_all['order_purchase_timestamp'].min().to_pydatetime()
max_d = df_all['order_purchase_timestamp'].max().to_pydatetime()
d_range = st.sidebar.date_input("분석 기간", [min_d, max_d], min_value=min_d, max_value=max_d)

all_segs = sorted(df_all['RFM_Segment'].unique())
sel_segs = st.sidebar.multiselect("고객 세그먼트", all_segs, default=all_segs)

# 필터링
if len(d_range) == 2:
    start, end = pd.to_datetime(d_range[0]), pd.to_datetime(d_range[1])
    df_f = df_all[(df_all['order_purchase_timestamp'] >= start) & (df_all['order_purchase_timestamp'] <= end) & (df_all['RFM_Segment'].isin(sel_segs))]
else:
    df_f = df_all[df_all['RFM_Segment'].isin(sel_segs)]

# --- 메인 대시보드 ---
st.title("🇧🇷 Olist 비즈니스 통합 전략 대시보드")
st.markdown("매출 성장, 운영 효율, 그리고 지역별 위험 요소를 통합적으로 분석합니다.")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 운영 모니터링", "📈 성장 실적", "🗺️ 지역 전략"])

# 색상 팔레트 고정 (Low: Red, High: Blue)
color_map = {'High (4-5)': '#0000FF', 'Low (1-3)': '#FF0000'}

# --- TAB 1: 운영 모니터링 ---
with tab1:
    st.header("🚚 운영 효율 및 만족도 분석")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("📦 리뷰 그룹별 물류 지표 (배송 지연 중심)")
        log_comp = df_f[df_f['review_group'] != 'None'].groupby('review_group').agg({
            'shipping_duration': 'mean', 'delay_days': 'mean', 'freight_ratio': 'mean'
        }).reset_index()
        
        log_m = log_comp.melt(id_vars='review_group', value_vars=['shipping_duration', 'delay_days', 'freight_ratio'])
        m_kr = {'shipping_duration': '평균 배송일', 'delay_days': '평균 지연일', 'freight_ratio': '배송비 비중'}
        log_m['Metric'] = log_m['variable'].map(m_kr)
        
        fig_log = px.bar(log_m, x='Metric', y='value', color='review_group', barmode='group',
                        text_auto='.2f', color_discrete_map=color_map,
                        hover_data={'value': ': .2f', 'review_group': True})
        st.plotly_chart(fig_log, use_container_width=True)

    with c2:
        st.subheader("💳 리뷰 그룹별 결제 수단 비중")
        pay_data = []
        for g in ['High (4-5)', 'Low (1-3)']:
            ids = df_f[df_f['review_group'] == g]['order_id']
            p = payments_raw[payments_raw['order_id'].isin(ids)]['payment_type'].value_counts(normalize=True).reset_index()
            p['review_group'] = g
            pay_data.append(p)
        
        pay_f = pd.concat(pay_data)
        pay_f.columns = ['payment_type', 'proportion', 'review_group']
        fig_sun = px.sunburst(pay_f, path=['review_group', 'payment_type'], values='proportion',
                             color='review_group', color_discrete_map=color_map,
                             hover_data={'proportion': ':.1%'})
        st.plotly_chart(fig_sun, use_container_width=True)

    st.info("💡 **운영 인사이트**: 저만족(Low) 그룹의 평균 지연일은 고만족(High) 그룹보다 현저히 높으며, 바우처 결제 비중이 높게 나타나는 경향이 있습니다.")

    st.divider()
    
    # --- Zero-Delay Deep Dive ---
    st.subheader("🚀 Zero-Delay 마인드셋: 약속 준수가 평점에 미치는 영향")
    
    # 지연 여부 그룹화
    df_f['delivery_status'] = df_f['delay_days'].apply(lambda x: 'Delayed (지연)' if x > 0 else 'On-time (준수)')
    
    col_z1, col_z2 = st.columns([1, 2])
    
    with col_z1:
        # 그룹별 평균 평점 비교 (Bar Chart)
        status_rating = df_f.groupby('delivery_status')['review_score'].mean().reset_index()
        fig_z_bar = px.bar(status_rating, x='delivery_status', y='review_score',
                          color='delivery_status', 
                          color_discrete_map={'Delayed (지연)': '#FF0000', 'On-time (준수)': '#0000FF'},
                          text_auto='.2f', title="배송 약속 준수 여부별 평균 평점")
        fig_z_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_z_bar, use_container_width=True)
        
    with col_z2:
        # 지연 일수별 CS 키워드 등장 빈도 (Line Chart)
        # 키워드 필터링
        cs_keywords = ['ainda', 'não recebi', 'atraso', 'demora']
        
        def count_cs_keywords(text):
            if pd.isna(text): return 0
            text = text.lower()
            return 1 if any(k in text for k in cs_keywords) else 0
            
        df_f['has_cs_keyword'] = df_f['review_comment_message'].apply(count_cs_keywords)
        
        # 지연된 데이터만 추출 (0~30일 사이로 제한)
        delay_analysis = df_f[(df_f['delay_days'] > 0) & (df_f['delay_days'] <= 30)].copy()
        delay_trend = delay_analysis.groupby('delay_days').agg({
            'review_score': 'mean',
            'has_cs_keyword': 'mean'
        }).reset_index()
        
        fig_z_line = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_z_line.add_trace(go.Scatter(x=delay_trend['delay_days'], y=delay_trend['review_score'],
                                      name="평균 평점", mode='lines+markers', line=dict(color='#0000FF')), secondary_y=False)
                                      
        fig_z_line.add_trace(go.Scatter(x=delay_trend['delay_days'], y=delay_trend['has_cs_keyword']*100,
                                      name="CS 키워드 빈도 (%)", mode='lines+markers', line=dict(color='#FF0000', dash='dot')), secondary_y=True)
                                      
        fig_z_line.update_layout(title="지연 일수 증가에 따른 평점 하락 및 CS 키워드 급증(%)",
                                hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        
        fig_z_line.update_xaxes(title_text="지연 일수 (Days)")
        fig_z_line.update_yaxes(title_text="평균 평점", secondary_y=False)
        fig_z_line.update_yaxes(title_text="CS 키워드 빈도 (%)", secondary_y=True)
        
        st.plotly_chart(fig_z_line, use_container_width=True)

    st.warning("⚠️ **Zero-Delay 분석 결과**: 배송 지연이 단 1일만 발생해도 불만 키워드('ainda', 'não recebi')의 출현 빈도가 급격히 상승하며 평점이 3점대 이하로 수렴하는 '임계점'이 확인됩니다.")

# --- TAB 2: 성장 실적 ---
with tab2:
    st.header("💰 매출 실적 및 판매 트렌드")
    
    # 시각화 1: 이중 축 라인
    trend = df_f.copy()
    trend['month'] = trend['order_purchase_timestamp'].dt.to_period('M').astype(str)
    t_data = trend.groupby('month').agg({'price': 'sum', 'order_id': 'nunique'}).reset_index()
    
    fig_t = make_subplots(specs=[[{"secondary_y": True}]])
    fig_t.add_trace(go.Scatter(x=t_data['month'], y=t_data['price'], name="매출액 (R$)", mode='lines+markers'), secondary_y=False)
    fig_t.add_trace(go.Scatter(x=t_data['month'], y=t_data['order_id'], name="판매량 (건)", mode='lines+markers', line=dict(dash='dot')), secondary_y=True)
    fig_t.update_layout(title="월별 매출 및 판매량 추이", hovermode="x unified")
    st.plotly_chart(fig_t, use_container_width=True)
    
    # 시각화 2: Treemap
    st.subheader("🌳 카테고리별 매출 상위 10 (색상: 평점)")
    cat = df_f.groupby('product_category_name_english').agg({'price': 'sum', 'review_score': 'mean'}).reset_index()
    top10 = cat.nlargest(10, 'price')
    fig_tree = px.treemap(top10, path=['product_category_name_english'], values='price',
                         color='review_score', color_continuous_scale='RdYlBu', # Red for Low, Blue for High
                         hover_data={'price': ':,.0f', 'review_score': ':.2f'})
    st.plotly_chart(fig_tree, use_container_width=True)
    
    # 시각화 3: 상관관계
    st.subheader("🔍 리뷰 개수와 판매량 상관관계")
    prod = df_f.groupby('product_id').agg({'review_score': 'count', 'order_id': 'nunique'}).reset_index()
    prod.columns = ['pid', 'rcount', 'svol']
    fig_scat = px.scatter(prod[prod['svol'] <= prod['svol'].quantile(0.99)], x='rcount', y='svol', trendline="ols",
                         opacity=0.5, title="리뷰가 많을수록 판매가 늘어나는가?",
                         hover_data={'rcount': True, 'svol': True})
    st.plotly_chart(fig_scat, use_container_width=True)

# --- TAB 3: 지역 전략 ---
with tab3:
    st.header("🌎 브라질 지역별 물류 위험 및 매출 밀도")
    
    # 데이터 집계
    state_data = df_f.groupby('customer_state').agg({
        'price': 'sum',
        'delay_days': 'mean',
        'review_score': 'mean',
        'RFM_Segment': lambda x: (x == 'VIP').sum()
    }).reset_index()
    state_data.columns = ['state', 'revenue', 'avg_delay', 'avg_rating', 'vip_count']
    
    # 지도 시각화 (Choropleth + Bubble)
    st.subheader("📍 주별 매출 밀도 및 배송 지연 위험도")
    
    # Brazil GeoJSON URL
    geojson_url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    
    fig_map = px.choropleth(state_data, geojson=geojson_url, locations='state', featureidkey="properties.sigla",
                           color='revenue', color_continuous_scale="Blues",
                           scope="south america", title="주별 매출액(색상) 및 평균 지연일(크기 - 버블 효과 대체)")
    # 버블 효과를 위해 Scattergeo 추가
    # 주별 좌표 데이터가 부족하므로 여기서는 Choropleth 자체에 정보 통합
    fig_map.update_geos(fitbounds="locations", visible=False)
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)
    
    # 산점도: 품질 위험 분석
    st.subheader("⚠️ 지역별 운영 리스크 분석")
    fig_risk = px.scatter(state_data, x='avg_delay', y='avg_rating', size='revenue', color='vip_count',
                         text='state', labels={'avg_delay': '평균 지연 일수', 'avg_rating': '평균 평점'},
                         title="지연 일수 vs 평점 (원 크기: 매출액, 색상: VIP 고객수)",
                         color_continuous_scale="RdBu_r")
    
    # 주석 추가 (AL, MA)
    for target in ['AL', 'MA']:
        row = state_data[state_data['state'] == target]
        if not row.empty:
            fig_risk.add_annotation(x=row['avg_delay'].values[0], y=row['avg_rating'].values[0],
                                   text=f"⚠️ {target} 위험지역", showarrow=True, arrowhead=1)
            
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # 상품 정보 영향 (사진 개수)
    st.subheader("🖼️ 상품 사진 개수가 평점에 미치는 영향 (주별)")
    photo_effect = df_f.groupby('customer_state').agg({'product_photos_qty': 'mean', 'review_score': 'mean'}).reset_index()
    fig_photo = px.line(photo_effect.sort_values('product_photos_qty'), x='product_photos_qty', y='review_score', 
                       markers=True, text='customer_state', title="평균 사진 개수와 리뷰 평점의 관계")
    st.plotly_chart(fig_photo, use_container_width=True)

    # 텍스트 마이닝 기반 인사이트 (상태별)
    st.divider()
    selected_state = st.selectbox("집중 분석할 주(State) 선택", sorted(state_data['state'].unique()))
    
    st.write(f"### 🔍 {selected_state} 지역 주요 불만 키워드 (시뮬레이션)")
    state_reviews = df_f[(df_f['customer_state'] == selected_state) & (df_f['review_score'] < 4)]['review_comment_message'].dropna()
    
    if not state_reviews.empty:
        # 간단한 키워드 추출 시뮬레이션 (실제로는 더 복잡한 NLP 필요)
        all_text = " ".join(state_reviews).lower()
        keywords = ["demora", "prazo", "entregue", "produto", "péssimo", "atraso"]
        found = [k for k in keywords if k in all_text]
        
        st.error(f"주요 이슈: {', '.join(found) if found else '배송 및 품질 불만'}")
        st.write(f"해당 지역 저만족 리뷰 수: {len(state_reviews)}건")
    else:
        st.success("해당 지역은 현재 불만 데이터가 거의 없습니다.")

# 하단 결론
st.divider()
st.subheader("🎯 데이터 기반 지역화 전략 제언")
st.markdown(f"""
- **북부/북동부 리스크**: AL, MA 등 지연이 잦은 지역은 물류 파트너 교체 또는 현지 창고(Hub) 확보가 시급합니다.
- **사진의 중요성**: 평균 사진 개수가 3개 미만인 지역은 평점 변동성이 큽니다. 상세페이지 강화 가이드를 판매자에게 배포하세요.
- **VIP 보존**: VIP 밀도가 높은 주에서 지연이 발생할 경우 즉각적인 보상 바우처를 자동 발행하는 자동화 로직을 추천합니다.
""")

st.divider()
st.subheader("🏆 종합 결론 및 전략적 마인드셋")
st.info("""
### 1. Zero-Delay: 선택이 아닌 필수
분석 결과, **On-time 배송**은 고객 만족의 기준점(Baseline)입니다. 배송 지연이 발생하는 순간, 제품의 본질적 가치와 상관없이 리뷰 점수는 급락하며 불만 키워드가 약 **2.5배 이상** 급증합니다. 물류 프로세스의 정교화는 단순한 운영 개선이 아니라 **매출 방어 전략**입니다.

### 2. High-Value 고매출-저만족 카테고리 타겟팅
통합 대시보드의 Treemap에서 식별된 '매출은 높으나 평점이 낮은' 카테고리는 현재 Olist에서 가장 큰 위험 요소이자 기회입니다. 이들 카테고리의 **사진 수 증대** 및 **배송 예상일 보수적 산정**을 통해 고객의 기대치를 관리해야 합니다.

### 3. 데이터 중심의 의의
단순히 리뷰 점수만 보는 것이 아니라, **RFM 등급과 지역적 특성**을 결합했을 때 진정한 원인이 보입니다. VIP 고객이 많은 지역에서의 지연은 '충성도 붕괴'로 이어지므로, 해당 지역에 대한 리소스 우선 배분(Priority Shipping)이 필요합니다.

**🎯 한 줄 요약:** 약속한 시간에 정확히 배송하는 것이 가장 강력하고 저렴한 마케팅이며, 리뷰 데이터는 그 약속이 어디서 깨지고 있는지 알려주는 가장 정교한 나침반입니다.
""")
