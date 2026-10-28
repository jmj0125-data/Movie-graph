import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 스크린 규모, 관객 수, 10위권 유지 기간, 롱런 지수를 바탕으로 비슷한 영화끼리 묶습니다.")

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

FEATURES = {
    "스크린 수(로그)": "first_scrn_log",
    "누적 관객(로그)": "total_audi_log",
    "10위권 일수": "days_in_top10",
    "롱런 지수": "long_run_index",
}
RAW_COLUMNS = {
    "스크린 수 평균": "first_scrn",
    "누적 관객 평균": "total_audi",
    "10위권 일수 평균": "days_in_top10",
    "롱런 지수 평균": "long_run_index",
}
LABELS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
COLORS = ["#E45756", "#4C78A8", "#54A24B", "#F2CF5B", "#B279A2", "#FF9DA6", "#9D755D"]


@st.cache_data
def load_data(url):
    df = pd.read_csv(url, encoding="utf-8")
    expected = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError("필요한 열이 없습니다: " + ", ".join(missing))

    for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["long_run_index"] = np.where(
        df["first_week_audi"] > 0,
        (df["total_audi"] / df["first_week_audi"]).clip(upper=20),
        np.nan
    )
    df["first_scrn_log"] = np.where(
        df["first_scrn"] > 0, np.log10(df["first_scrn"]), np.nan
    )
    df["total_audi_log"] = np.where(
        df["total_audi"] > 0, np.log10(df["total_audi"]), np.nan
    )
    return df


try:
    source_df = load_data(DATA_URL)
except Exception as e:
    st.error("데이터를 불러오지 못했습니다. 데이터 주소와 CSV 열 이름을 확인해 주세요.")
    st.exception(e)
    st.stop()

# 모든 속성 중 결측이 있거나 첫 주 관객이 0 이하인 영화는 제외
valid_mask = (
    source_df["first_week_audi"].notna()
    & (source_df["first_week_audi"] > 0)
    & source_df[list(FEATURES.values())].notna().all(axis=1)
    & source_df["movieNm"].notna()
)
valid_df = source_df.loc[valid_mask].copy()

st.markdown(
    f"**전체 영화 {len(source_df):,}편 · 묶을 수 있는 영화 {len(valid_df):,}편** "
    "(필요한 값이 없거나 첫 주 관객이 0 이하인 영화 제외)"
)

# -----------------------------
# 속성 및 묶음 수 선택
# -----------------------------
st.subheader("1. 묶는 데 사용할 속성과 묶음 수")
left, right = st.columns([2, 1])
with left:
    selected_features = st.multiselect(
        "군집화에 사용할 속성(두 개 이상 선택)",
        options=list(FEATURES.keys()),
        default=list(FEATURES.keys()),
        help="선택한 속성만 표준화한 뒤 K-평균 군집화에 사용합니다."
    )
with right:
    n_clusters = st.selectbox(
        "묶음 수",
        options=list(range(2, 8)),
        index=1,
        help="2개부터 7개까지 선택할 수 있습니다."
    )

if len(selected_features) < 2:
    st.warning("K-평균 군집화를 하려면 속성을 두 개 이상 선택하세요.")
    st.stop()

if len(valid_df) < n_clusters:
    st.error(f"유효한 영화가 {n_clusters}편 미만이라 {n_clusters}개 묶음으로 나눌 수 없습니다.")
    st.stop()

feature_cols = [FEATURES[name] for name in selected_features]
scaler = StandardScaler()
scaled_values = scaler.fit_transform(valid_df[feature_cols])

# 현재 선택한 묶음 수로 모델 학습
model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
valid_df["cluster_raw"] = model.fit_predict(scaled_values)

# 누적 관객 평균이 큰 묶음부터 기호를 배정
raw_order = (
    valid_df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
raw_to_label = {raw: LABELS[i] for i, raw in enumerate(raw_order)}
valid_df["유형"] = valid_df["cluster_raw"].map(raw_to_label)
label_order = LABELS[:n_clusters]
palette = {LABELS[i]: COLORS[i] for i in range(n_clusters)}
valid_df["유형"] = pd.Categorical(valid_df["유형"], categories=label_order, ordered=True)

# 선택 속성으로 k=1..7의 군집 내 제곱거리 합 계산
# KMeans의 inertia_는 각 점에서 해당 군집 중심까지의 유클리드 거리 제곱 합
inertia_rows = []
for k in range(1, 8):
    if k == 1:
        km = KMeans(n_clusters=1, random_state=42, n_init=10)
    else:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(scaled_values)
    inertia_rows.append({"묶음 수": k, "중심까지 거리 제곱 합": float(km.inertia_)})

inertia_df = pd.DataFrame(inertia_rows)
inertia_df["직전 값보다 감소한 양"] = (
    inertia_df["중심까지 거리 제곱 합"].shift(1)
    - inertia_df["중심까지 거리 제곱 합"]
)
inertia_df.loc[0, "직전 값보다 감소한 양"] = np.nan

# 현재 묶음 수의 실루엣 점수
if 2 <= n_clusters < len(valid_df) and valid_df["cluster_raw"].nunique() > 1:
    sil_score = silhouette_score(scaled_values, valid_df["cluster_raw"])
else:
    sil_score = np.nan

# -----------------------------
# 2차원 산점도
# -----------------------------
st.subheader("2. 2차원 산점도")
axis1, axis2 = st.columns(2)
with axis1:
    x_feature = st.selectbox("가로축 속성", selected_features, index=0, key="scatter_x")
with axis2:
    y_feature = st.selectbox(
        "세로축 속성",
        selected_features,
        index=min(1, len(selected_features) - 1),
        key="scatter_y"
    )

fig2 = px.scatter(
    valid_df,
    x=FEATURES[x_feature],
    y=FEATURES[y_feature],
    color="유형",
    category_orders={"유형": label_order},
    color_discrete_map=palette,
    hover_name="movieNm",
    hover_data={
        "유형": True,
        FEATURES[x_feature]: ":.3f",
        FEATURES[y_feature]: ":.3f",
        "movieNm": False,
    },
    labels={
        FEATURES[x_feature]: x_feature,
        FEATURES[y_feature]: y_feature,
        "유형": "영화 유형",
    },
)
fig2.update_traces(marker={"size": 8, "opacity": 0.78})
fig2.update_layout(margin=dict(l=10, r=10, t=25, b=10), height=520, legend_title_text="영화 유형")
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 3차원 산점도
# -----------------------------
st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 묶는 데 사용할 속성을 세 개 이상 선택하세요.")
else:
    xyz = st.columns(3)
    with xyz[0]:
        x3 = st.selectbox("X축 속성", selected_features, index=0, key="scatter3_x")
    with xyz[1]:
        y3 = st.selectbox("Y축 속성", selected_features, index=1, key="scatter3_y")
    with xyz[2]:
        z3 = st.selectbox("Z축 속성", selected_features, index=2, key="scatter3_z")

    fig3 = px.scatter_3d(
        valid_df,
        x=FEATURES[x3],
        y=FEATURES[y3],
        z=FEATURES[z3],
        color="유형",
        category_orders={"유형": label_order},
        color_discrete_map=palette,
        hover_name="movieNm",
        hover_data={
            "유형": True,
            FEATURES[x3]: ":.3f",
            FEATURES[y3]: ":.3f",
            FEATURES[z3]: ":.3f",
            "movieNm": False,
        },
        labels={
            FEATURES[x3]: x3,
            FEATURES[y3]: y3,
            FEATURES[z3]: z3,
            "유형": "영화 유형",
        },
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=650,
        legend_title_text="영화 유형",
        scene={"xaxis_title": x3, "yaxis_title": y3, "zaxis_title": z3},
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("마우스로 드래그하면 시점을 회전할 수 있습니다. 점에 마우스를 올리면 영화 제목이 표시됩니다.")

# -----------------------------
# 유형별 요약표
# -----------------------------
st.subheader("4. 유형별 편수와 속성 평균")
summary_rows = []
for label in label_order:
    group = valid_df[valid_df["유형"] == label]
    if group.empty:
        continue
    summary_rows.append({
        "유형": label,
        "편수": len(group),
        "스크린 수 평균": group["first_scrn"].mean(),
        "누적 관객 평균": group["total_audi"].mean(),
        "10위권 일수 평균": group["days_in_top10"].mean(),
        "롱런 지수 평균": group["long_run_index"].mean(),
    })

summary_df = pd.DataFrame(summary_rows)
st.dataframe(
    summary_df.style.format({
        "편수": "{:,.0f}",
        "스크린 수 평균": "{:,.1f}",
        "누적 관객 평균": "{:,.0f}",
        "10위권 일수 평균": "{:,.2f}",
        "롱런 지수 평균": "{:,.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# 유형별 누적 관객 TOP 5
# -----------------------------
st.subheader("5. 유형별 누적 관객 TOP 5")
top_cols = st.columns(min(4, n_clusters))
for i, label in enumerate(label_order):
    group = valid_df[valid_df["유형"] == label].sort_values("total_audi", ascending=False).head(5)
    with top_cols[i % len(top_cols)]:
        st.markdown(f"**{label} 유형**")
        if group.empty:
            st.write("해당 유형에 영화가 없습니다.")
        else:
            top_table = group[["movieNm", "total_audi"]].rename(
                columns={"movieNm": "영화 제목", "total_audi": "누적 관객"}
            ).reset_index(drop=True)
            top_table.index = top_table.index + 1
            st.dataframe(
                top_table.style.format({"누적 관객": "{:,.0f}"}),
                use_container_width=True,
            )

# -----------------------------
# 엘보 그래프와 표
# -----------------------------
st.subheader("6. 묶음 수에 따른 중심까지 거리 제곱 합")
st.write("값은 각 영화 점에서 자신이 속한 묶음의 중심까지의 거리 제곱을 모두 더한 것입니다. 값이 작을수록 각 묶음 안의 점들이 중심에 가깝습니다.")

fig_inertia = go.Figure()
fig_inertia.add_trace(go.Scatter(
    x=inertia_df["묶음 수"],
    y=inertia_df["중심까지 거리 제곱 합"],
    mode="lines+markers",
    name="거리 제곱 합",
    line=dict(width=3),
    marker=dict(size=8),
))
current_inertia = float(inertia_df.loc[inertia_df["묶음 수"] == n_clusters, "중심까지 거리 제곱 합"].iloc[0])
fig_inertia.add_vline(
    x=n_clusters,
    line_dash="dash",
    line_width=2,
    annotation_text=f"현재 묶음 수: {n_clusters}",
    annotation_position="top",
)
fig_inertia.update_layout(
    xaxis=dict(title="묶음 수", tickmode="linear", dtick=1),
    yaxis_title="중심까지 거리 제곱 합",
    margin=dict(l=10, r=10, t=50, b=10),
    height=430,
    showlegend=False,
)
st.plotly_chart(fig_inertia, use_container_width=True)

st.markdown("**묶음 수별 거리 제곱 합과 직전 값 대비 감소량**")
st.dataframe(
    inertia_df.style.format({
        "묶음 수": "{:.0f}",
        "중심까지 거리 제곱 합": "{:,.3f}",
        "직전 값보다 감소한 양": "{:,.3f}",
    }, na_rep=""),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# 실루엣 점수
# -----------------------------
st.subheader("7. 현재 묶음 수의 실루엣 점수")
if np.isfinite(sil_score):
    st.markdown(
        f"**현재 묶음 수 {n_clusters}개의 실루엣 점수: {sil_score:.4f}**"
    )
    st.caption("실루엣 점수는 -1에서 1 사이입니다. 1에 가까울수록 각 영화가 자기 유형에 잘 속하고 다른 유형과 구분되는 경향이 큽니다.")
else:
    st.info("현재 데이터로 실루엣 점수를 계산할 수 없습니다.")

with st.expander("분석 방법"):
    st.markdown(
        "- 스크린 수와 누적 관객은 `log10` 상용로그로 변환합니다.\n"
        "- 롱런 지수는 `누적 관객 ÷ 첫 주 관객`이며, 20을 넘으면 20으로 제한합니다.\n"
        "- 필요한 값이 없거나 첫 주 관객이 0 이하인 영화는 분석에서 제외합니다.\n"
        "- 선택한 속성만 표준화하여 K-평균 군집화에 사용합니다. 난수 시드는 42로 고정합니다.\n"
        "- 유형 기호는 누적 관객 평균이 높은 순서대로 ㉮부터 배정합니다.\n"
        "- 유형별 평균 표는 로그 변환 전의 원래 단위로 계산합니다.\n"
        "- 거리 제곱 합은 현재 선택한 속성으로 표준화한 자료에서 묶음 수 1~7 각각에 대해 계산합니다."
    )
