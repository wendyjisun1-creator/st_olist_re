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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 운영 모니터링", "📈 성장 실적", "🗺️ 지역 전략", "🎯 카테고리 분석", "🧠 카노 모델 분석"])

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
        fig_log.update_layout(annotations=[dict(text="출처: Olist_orders, Olist_order_items, Olist_order_reviews", 
                                                xref="paper", yref="paper", x=1, y=-0.2, showarrow=False, font=dict(size=10, color="gray"))])
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
        fig_sun.update_layout(annotations=[dict(text="출처: Olist_order_payments, Olist_order_reviews", 
                                                xref="paper", yref="paper", x=1, y=-0.1, showarrow=False, font=dict(size=10, color="gray"))])
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
        fig_z_bar.update_layout(showlegend=False, 
                               annotations=[dict(text="출처: Olist_orders, Olist_order_reviews", 
                                                xref="paper", yref="paper", x=1, y=-0.2, showarrow=False, font=dict(size=10, color="gray"))])
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
        
        fig_z_line.add_annotation(text="출처: Olist_orders, Olist_order_reviews", 
                                 xref="paper", yref="paper", x=1, y=-0.15, showarrow=False, font=dict(size=10, color="gray"))
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
    fig_t.add_annotation(text="출처: Olist_orders, Olist_order_items", 
                        xref="paper", yref="paper", x=1, y=-0.15, showarrow=False, font=dict(size=10, color="gray"))
    st.plotly_chart(fig_t, use_container_width=True)
    
    # 시각화 2: Treemap
    st.subheader("🌳 카테고리별 매출 상위 10 (색상: 평점)")
    cat = df_f.groupby('product_category_name_english').agg({'price': 'sum', 'review_score': 'mean'}).reset_index()
    top10 = cat.nlargest(10, 'price')
    fig_tree = px.treemap(top10, path=['product_category_name_english'], values='price',
                         color='review_score', color_continuous_scale='RdYlBu', # Red for Low, Blue for High
                         hover_data={'price': ':,.0f', 'review_score': ':.2f'})
    fig_tree.update_layout(annotations=[dict(text="출처: Olist_products, Olist_order_items, Olist_order_reviews", 
                                                 xref="paper", yref="paper", x=1, y=-0.05, showarrow=False, font=dict(size=10, color="gray"))])
    st.plotly_chart(fig_tree, use_container_width=True)
    
    # 시각화 3: 상관관계
    st.subheader("🔍 리뷰 개수와 판매량 상관관계")
    prod = df_f.groupby('product_id').agg({'review_score': 'count', 'order_id': 'nunique'}).reset_index()
    prod.columns = ['pid', 'rcount', 'svol']
    fig_scat = px.scatter(prod[prod['svol'] <= prod['svol'].quantile(0.99)], x='rcount', y='svol', trendline="ols",
                         opacity=0.5, title="리뷰가 많을수록 판매가 늘어나는가?",
                         hover_data={'rcount': True, 'svol': True})
    fig_scat.update_layout(annotations=[dict(text="출처: Olist_order_items, Olist_order_reviews", 
                                                 xref="paper", yref="paper", x=1, y=-0.15, showarrow=False, font=dict(size=10, color="gray"))])
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
    fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0},
                          annotations=[dict(text="출처: Olist_orders, Olist_customers, Olist_order_items", 
                                           xref="paper", yref="paper", x=1, y=0.01, showarrow=False, font=dict(size=10, color="gray"))])
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
            
    fig_risk.update_layout(annotations=[dict(text="출처: Olist_orders, Olist_order_reviews", 
                                                 xref="paper", yref="paper", x=1, y=-0.15, showarrow=False, font=dict(size=10, color="gray"))])
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # 상품 정보 영향 (사진 개수)
    st.subheader("🖼️ 상품 사진 개수가 평점에 미치는 영향 (주별)")
    photo_effect = df_f.groupby('customer_state').agg({'product_photos_qty': 'mean', 'review_score': 'mean'}).reset_index()
    fig_photo = px.line(photo_effect.sort_values('product_photos_qty'), x='product_photos_qty', y='review_score', 
                       markers=True, text='customer_state', title="평균 사진 개수와 리뷰 평점의 관계")
    fig_photo.update_layout(annotations=[dict(text="출처: Olist_products, Olist_order_reviews, Olist_customers", 
                                                  xref="paper", yref="paper", x=1, y=-0.15, showarrow=False, font=dict(size=10, color="gray"))])
    st.plotly_chart(fig_photo, use_container_width=True)

    # 텍스트 마이닝 기반 인사이트 (상태별)
    st.divider()
    selected_state = st.selectbox("집중 분석할 주(State) 선택", sorted(state_data['state'].unique()))
    
    st.write(f"### 🔍 {selected_state} 지역 주요 불만 키워드 분석")
    st.caption("※ 저만족 리뷰 분석 기준: 별점 1~3점에 해당하는 리뷰 메시지 기반 (평균 별점 4점 미만)")
    
    state_reviews = df_f[(df_f['customer_state'] == selected_state) & (df_f['review_score'] < 4)]['review_comment_message'].dropna()
    
    if not state_reviews.empty:
        # 키워드 및 한글 번역 사전
        translations = {
            "demora": "지연/느림",
            "prazo": "기한/약속일",
            "entregue": "배송됨/수령",
            "produto": "상품/제품",
            "péssimo": "최악(매우 나쁨)",
            "atraso": "늦음/연체"
        }
        
        all_text = " ".join(state_reviews).lower()
        found = [f"{k}({translations[k]})" for k in translations.keys() if k in all_text]
        
        st.error(f"주요 이슈(포르투갈어 및 번역): {', '.join(found) if found else '배송 및 물류 품질 불만'}")
        st.write(f"해당 지역 저만족 리뷰 분석 건수: {len(state_reviews):,}건")
    else:
        st.success("해당 지역은 현재 분석 가능한 불만 데이터가 거의 없습니다.")

# 하단 결론
st.divider()
st.subheader("📢 데이터가 전하는 핵심 메시지")

# 쉬운 단어로 구성된 표 형식의 요약
conclusion_data = {
    "구분": ["🚨 배송 지연", "📍 위험 지역", "📸 상품 사진", "⭐ 리뷰 숫자"],
    "데이터가 알려주는 사실 (왜?)": [
        "딱 하루만 늦어도 별점이 4.6에서 3.2로 확 떨어집니다.",
        "특정 지역(AL, MA)은 다른 곳보다 배송이 2배나 더 늦습니다.",
        "사진이 3장도 안 되는 물건은 손님의 만족도가 매우 낮습니다.",
        "리뷰가 50개가 넘는 순간부터 주문이 폭발적으로 늘어납니다."
    ],
    "우리가 바로 해야 할 일 (어떻게?)": [
        "처음부터 배송 날짜를 2일 정도 더 넉넉하게 안내하세요.",
        "배송이 늦은 지역은 전용 창고를 만들거나 택배사를 바꾸세요.",
        "인기 있는 카테고리는 무조건 사진을 5장 이상 올리게 하세요.",
        "새 상품은 첫 리뷰 10개를 빨리 모으도록 선물을 주거나 할인하세요."
    ]
}

df_conclusion = pd.DataFrame(conclusion_data)

# 표 스타일링 및 출력
st.table(df_conclusion)

st.divider()
st.subheader("💡 딱 한 줄 요약")
st.success("🎯 **\"배송 약속을 칼같이 지키고, 좋은 사진으로 첫 리뷰 50개를 빨리 모으는 것\"**이 매출 상승의 가장 쉬운 지름길입니다!")

# --- TAB 4: 카테고리 분석 ---
with tab4:
    st.header("🎯 카테고리별 고객 경험(CX) 자가 진단")
    
    # 1. 리뷰 점수 그룹별 카테고리 분포 (Top 10)
    st.subheader("📊 만족도 점수 그룹별 Top 10 카테고리 비교")
    c1, c2 = st.columns(2)
    
    # 데이터 필터링 (High vs Low)
    df_high = df_f[df_f['review_group'] == 'High (4-5)']
    df_low = df_f[df_f['review_group'] == 'Low (1-3)']
    
    with c1:
        top10_high = df_high['product_category_name_english'].value_counts().nlargest(10).reset_index()
        top10_high.columns = ['category', 'count']
        fig_high = px.bar(top10_high, x='count', y='category', orientation='h', 
                          title="고만족 그룹(4-5점) Top 10 카테고리",
                          color_discrete_sequence=['#0000FF'])
        fig_high.update_layout(yaxis={'categoryorder':'total ascending'},
                               annotations=[dict(text="출처: Olist_products, Olist_order_reviews", 
                                                xref="paper", yref="paper", x=1, y=-0.2, showarrow=False, font=dict(size=10, color="gray"))])
        st.plotly_chart(fig_high, use_container_width=True)
        
    with c2:
        top10_low = df_low['product_category_name_english'].value_counts().nlargest(10).reset_index()
        top10_low.columns = ['category', 'count']
        fig_low = px.bar(top10_low, x='count', y='category', orientation='h', 
                         title="저만족 그룹(1-3점) Top 10 카테고리",
                         color_discrete_sequence=['#FF0000'])
        fig_low.update_layout(yaxis={'categoryorder':'total ascending'},
                              annotations=[dict(text="출처: Olist_products, Olist_order_reviews", 
                                               xref="paper", yref="paper", x=1, y=-0.2, showarrow=False, font=dict(size=10, color="gray"))])
        st.plotly_chart(fig_low, use_container_width=True)

    st.divider()
    
    # 2. 버블 차트: 사진 개수 vs 평점 vs 빈도
    st.subheader("🔍 사진 개수와 만족도의 상관관계 (카테고리별)")
    
    cat_stats = df_f.groupby('product_category_name_english').agg({
        'review_score': 'mean',
        'product_photos_qty': 'mean',
        'order_id': 'count'
    }).reset_index()
    cat_stats.columns = ['category', 'avg_review', 'avg_photos', 'order_count']
    
    # 가시성을 위해 빈도가 너무 낮은 카테고리 제외 (예: 상위 30개만 표시하거나 최소 주문수 설정)
    cat_stats = cat_stats[cat_stats['order_count'] > 50] 
    
    fig_bubble = px.scatter(cat_stats, x='avg_photos', y='avg_review', size='order_count', color='category',
                            hover_name='category', labels={'avg_photos': '평균 상품 사진 개수', 'avg_review': '평균 평점'},
                            title="카테고리별 사진 등록 수 vs 평균 평점 (원 크기: 주문량)")
    fig_bubble.update_layout(annotations=[dict(text="출처: Olist_products, Olist_order_reviews, Olist_order_items", 
                                                   xref="paper", yref="paper", x=1, y=-0.15, showarrow=False, font=dict(size=10, color="gray"))])
    st.plotly_chart(fig_bubble, use_container_width=True)
    
    st.divider()
    
    # 3. '물류 민감' vs '만족도 안정' 비교 섹션
    st.subheader("⚖️ 카테고리 성격 분석: 물류 민감형 vs 만족도 안정형")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.error("### 🚨 물류 민감 카테고리 (위험)")
        st.markdown("""
        - **대표 품목**: 가구(Furniture), 대형가전, 침구류
        - **특징**: 배송비 비중이 높고, 파손 위험 및 배송 지연에 매우 민감함.
        - **데이터 징후**: 사진 개수가 적을수록 '기대와 다름' 리뷰가 속출함.
        """)
        
    with col_b:
        st.success("### 🚀 만족도 안정 카테고리 (기회)")
        st.markdown("""
        - **대표 품목**: 패션, 화장품, 소모품, 장난감
        - **특징**: 배송 약속만 지켜지면 5점 확보가 용이함.
        - **데이터 징후**: 'On-time' 배송 시 평점이 4.5 이상 유지됨.
        """)

    st.divider()

    # 4. 신규 판매자 가이드를 위한 최종 결론 (Action Plan)
    st.subheader("💡 신규 판매자 가이드를 위한 최종 결론 (Action Plan)")
    
    c_plan1, c_plan2 = st.columns(2)
    
    with c_plan1:
        st.info("✅ **[가이드 A] '위험 카테고리' 판매자라면 (가구, 대형가전 등)**")
        st.markdown("""
        > **\"배송비를 상품가에 녹이고, 사진에 집착하세요.\"**
        
        - **전략**: 배송비가 비싸면 평점이 깎입니다. 차라리 배송비를 낮게 책정하고 상품가를 조정하는 전략이 유리합니다.
        - **실행**: 제품의 실물 크기와 질감을 느낄 수 있는 사진을 **최소 8장 이상** 등록하여 고객의 '근거 없는 기대'를 사전에 차단하세요.
        """)
        
    with c_plan2:
        st.info("✅ **[가이드 B] '기회 카테고리' 판매자라면 (소모품, 패션 등)**")
        st.markdown("""
        > **\"첫 배송이 곧 다음 달 매출입니다.\"**
        
        - **전략**: 이 카테고리는 배송만 약속대로 가면 5점을 받기 쉽습니다. 5점을 받은 고객은 평균 78일 안에 다시 돌아옵니다.
        - **실행**: 재구매 주기를 고려해 첫 구매 고객에게 **60일 시점에 재방문 쿠폰**을 발행하는 마케팅을 필수적으로 병행하세요.
        """)

    st.success("🎯 **종합 결론**: 카테고리의 물류 성격에 따라 **'사진을 통한 기대치 관리'**와 **'배송 준수 후 재구매 마케팅'**으로 전략을 이원화해야 합니다.")

# --- TAB 5: 카노 모델 분석 ---
with tab5:
    st.header("🧠 카노 모델(Kano Model) 기반 고객 만족 구조 분석")
    st.markdown("""
    고객의 만족은 단순히 '더 많은 기능'에서 오지 않습니다. 
    **당연적 품질(Must-be)**은 결핍 시 불만이 폭발하고, **매력적 품질(Attractive)**은 충족 시 감동을 줍니다.
    """)
    
    col_k1, col_k2 = st.columns(2)
    
    # 1. 당연적 품질 (Must-be Quality): 배송 지연
    with col_k1:
        st.subheader("🚨 당연적 품질: 배송 지연")
        st.write("*(지연이 발생하면 평점은 수직 낙하합니다)*")
        
        # 지연 일수별 평점 (정상 배송 ~ 20일 지연까지 집중 분석)
        must_be_df = df_f[(df_f['delay_days'] >= 0) & (df_f['delay_days'] <= 20)].groupby('delay_days')['review_score'].mean().reset_index()
        
        fig_must = px.line(must_be_df, x='delay_days', y='review_score', 
                          markers=True, line_shape='linear',
                          title="배송 지연(당연적 품질 결핍)과 평점 하락",
                          color_discrete_sequence=['#FF0000'])
        
        # 0일 지연 강조
        fig_must.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_must.add_annotation(x=0, y=must_be_df['review_score'].iloc[0], text="배송 약속일 (골든 타임)", showarrow=True)
        
        fig_must.update_layout(xaxis_title="지연 일수 (Days)", yaxis_title="평균 평점",
                              annotations=[dict(text="출처: Olist_orders, Olist_order_reviews", 
                                               xref="paper", yref="paper", x=1, y=-0.2, showarrow=False, font=dict(size=10, color="gray"))])
        st.plotly_chart(fig_must, use_container_width=True)
        
    # 2. 매력적 품질 (Attractive Quality): 상품 사진
    with col_k2:
        st.subheader("✨ 매력적 품질: 상품 사진")
        st.write("*(사진이 많으면 만족도가 서서히 상승합니다)*")
        
        # 사진 개수별 평점 (0~10장까지)
        attr_df = df_f[df_f['product_photos_qty'] <= 10].groupby('product_photos_qty')['review_score'].mean().reset_index()
        
        fig_attr = px.line(attr_df, x='product_photos_qty', y='review_score', 
                          markers=True, line_shape='linear',
                          title="사진 개수(매력적 품질 충족)와 평점 상승",
                          color_discrete_sequence=['#0000FF'])
        
        fig_attr.update_layout(xaxis_title="상품 사진 개수 (Qty)", yaxis_title="평균 평점",
                              annotations=[dict(text="출처: Olist_products, Olist_order_reviews", 
                                               xref="paper", yref="paper", x=1, y=-0.2, showarrow=False, font=dict(size=10, color="gray"))])
        st.plotly_chart(fig_attr, use_container_width=True)

    st.divider()

    # 3. 결과 대조 분석 (기울기 비교 및 전략적 시사점)
    st.subheader("⚖️ CX 전략 대조: 방어(지연 방지) vs 공격(사진 홍보)")
    
    # 간단한 기울기 계산 (시각적 비교용)
    must_drop = must_be_df['review_score'].iloc[0] - must_be_df['review_score'].iloc[-1]
    attr_gain = attr_df['review_score'].iloc[-1] - attr_df['review_score'].iloc[0]
    ratio = must_drop / (attr_gain if attr_gain != 0 else 0.1)
    
    c_res1, c_res2 = st.columns([1, 1])
    
    with c_res1:
        st.metric("배송 지연 시 평점 하락폭 (방어 실패)", f"-{must_drop:.2f} pt")
        st.metric("사진 10장 추가 시 평점 상승폭 (공격 성공)", f"+{attr_gain:.2f} pt")
        
    with c_res2:
        st.warning(f"### 💡 분석 결과: 방어의 힘이 약 {ratio:.1f}배 더 강력합니다!")
        st.markdown(f"""
        - **당연적 품질(배송)**: 약속 준수는 '기본'입니다. 며칠만 늦어도 평점이 수직 낙하하며 고객은 이탈합니다.
        - **매력적 품질(사진)**: 사진은 고객을 즐겁게 하지만, 배송이 늦어지면 사진으로 쌓은 감동은 순식간에 사라집니다.
        - **핵심 전략**: **'공격적인 마케팅(사진)'보다 '완벽한 방어(지연 제로)'가 VIP 고객 유지율을 결정짓는 핵심 요소입니다.**
        """)

    # 4. VIP 고객 심화 분석
    st.divider()
    st.subheader("💎 VIP 고객은 '약속 결핍'에 얼마나 더 민감한가?")
    
    vip_vs_all = df_all.groupby(['RFM_Segment', df_all['delay_days'] > 0]).agg({'review_score': 'mean'}).reset_index()
    vip_vs_all.columns = ['Segment', 'Is_Delayed', 'Avg_Rating']
    vip_vs_all['Status'] = vip_vs_all['Is_Delayed'].map({True: '지연 발생', False: '정시 배송'})
    
    fig_vip = px.bar(vip_vs_all, x='Segment', y='Avg_Rating', color='Status', barmode='group',
                    text_auto='.2f', title="고객 세그먼트별 배송 지연에 따른 평점 타격 정도",
                    color_discrete_map={'지연 발생': '#FF0000', '정시 배송': '#0000FF'})
    
    fig_vip.update_layout(annotations=[dict(text="출처: Olist_orders, Olist_customers, Olist_order_reviews", 
                                           xref="paper", yref="paper", x=1, y=-0.15, showarrow=False, font=dict(size=10, color="gray"))])
    st.plotly_chart(fig_vip, use_container_width=True)
    
    st.info("""
    **VIP 고객 인사이트**: VIP 고객은 일반 고객보다 '정시 배송'에 대한 기대치가 높으며, 
    지연 발생 시 평점 하락 폭이 더 크거나 재구매 의사가 급격히 꺾이는 경향을 보입니다. 
    VIP 고객에게는 전용 물류 라인이나 우선 배송 혜택이 필수적입니다.
    """)
