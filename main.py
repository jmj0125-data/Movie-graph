import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"
FEATURES = ["로그 스크린 수", "로그 누적 관객", "10위권 일수", "롱런 지수"]
RAW_MEAN_COLUMNS = ["스크린 수", "누적 관객", "10위권 일수", "롱런 지수"]
CLUSTER_NAMES = ["㉮", "㉯", "㉰"]
SEED = 42

st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("영화의 개봉 초기 규모와 흥행 성과, 장기 상영 특성을 바탕으로 비슷한 영화끼리 묶습니다.")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    required = [
        "movieCd", "movieNm", "openDt", "genre", "nation",
        "first_scrn", "first_show", "first_date", "peak",
        "first_week_audi", "total_audi", "days_in_top10"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing))
    return df

try:
    original = load_data()
except Exception as e:
    st.error(f"데이터를 불러오지 못했습니다. 데이터 주소와 CSV 형식을 확인해 주세요.\n\n오류: {e}")
    st.stop()

df = original.copy()
for col in ["first_scrn", "first_week_audi", "total_audi", "days_in_top10"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 네 속성에 필요한 값이 모두 있고 첫 주 관객이 0보다 큰 영화만 사용
valid = (
    df["first_scrn"].notna()
    & df["total_audi"].notna()
    & df["days_in_top10"].notna()
    & df["first_week_audi"].notna()
    & (df["first_scrn"] > 0)
    & (df["total_audi"] > 0)
    & (df["first_week_audi"] > 0)
)
work = df.loc[valid].copy()
work["스크린 수"] = work["first_scrn"]
work["누적 관객"] = work["total_audi"]
work["10위권 일수"] = work["days_in_top10"]
work["롱런 지수"] = (work["total_audi"] / work["first_week_audi"]).clip(upper=20)
work["로그 스크린 수"] = np.log10(work["first_scrn"])
work["로그 누적 관객"] = np.log10(work["total_audi"])

# KMeans 입력 속성과 원래 단위 평균에 모두 유한한 값만 사용
numeric_cols = FEATURES + RAW_MEAN_COLUMNS
finite = np.isfinite(work[numeric_cols].to_numpy(dtype=float)).all(axis=1)
work = work.loc[finite].copy()

st.markdown(
    f"**전체 영화 {len(original):,}편 · 분석에 사용한 영화 {len(work):,}편**"
    f" (제외 {len(original) - len(work):,}편)"
)

if len(work) < 3:
    st.error("유효한 영화가 3편 미만이라 세 유형으로 나눌 수 없습니다.")
    st.stop()

st.subheader("1. 유형 분류에 사용할 속성")
selected_features = st.multiselect(
    "K-평균에 사용할 속성을 두 개 이상 선택하세요.",
    options=FEATURES,
    default=FEATURES,
    help="선택한 속성만 표준화해 영화 유형을 분류합니다."
)
if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 선택해 주세요.")
    st.stop()

X = work[selected_features].to_numpy(dtype=float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=SEED, n_init=10)
raw_labels = kmeans.fit_predict(X_scaled)
work = work.copy()
work["_cluster_raw"] = raw_labels

# 누적 관객의 원래 단위 평균이 높은 군집부터 ㉮, ㉯, ㉰ 부여
cluster_order = (
    work.groupby("_cluster_raw")["누적 관객"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)
name_map = {cluster_id: CLUSTER_NAMES[i] for i, cluster_id in enumerate(cluster_order)}
work["영화 유형"] = work["_cluster_raw"].map(name_map)
work["영화 제목"] = work["movieNm"].fillna("(제목 없음)").astype(str)

st.subheader("2. 2차원 산점도")
axis_options = selected_features
default_x = "로그 스크린 수" if "로그 스크린 수" in axis_options else axis_options[0]
default_y = next((f for f in axis_options if f != default_x), axis_options[1])
col_x, col_y = st.columns(2)
with col_x:
    x_axis = st.selectbox("가로축 속성", axis_options, index=axis_options.index(default_x), key="2d_x")
with col_y:
    y_axis = st.selectbox("세로축 속성", axis_options, index=axis_options.index(default_y), key="2d_y")

fig2 = px.scatter(
    work,
    x=x_axis,
    y=y_axis,
    color="영화 유형",
    hover_name="영화 제목",
    hover_data={"영화 유형": True, "movieCd": True, x_axis: ":.3f", y_axis: ":.3f"},
    category_orders={"영화 유형": CLUSTER_NAMES},
    color_discrete_sequence=px.colors.qualitative.Set2,
    labels={x_axis: x_axis, y_axis: y_axis},
    title=f"{x_axis} × {y_axis}"
)
fig2.update_traces(marker={"size": 8, "opacity": 0.8})
fig2.update_layout(legend_title_text="영화 유형", margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. 3차원 산점도")
if len(selected_features) < 3:
    st.info("3차원 산점도를 보려면 유형 분류 속성을 세 개 이상 선택해 주세요.")
else:
    three_defaults = []
    for candidate in ["로그 스크린 수", "로그 누적 관객", "10위권 일수"]:
        if candidate in selected_features and candidate not in three_defaults:
            three_defaults.append(candidate)
    for candidate in selected_features:
        if len(three_defaults) < 3 and candidate not in three_defaults:
            three_defaults.append(candidate)
    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox("X축 속성", selected_features, index=selected_features.index(three_defaults[0]), key="3d_x")
    with c2:
        y3 = st.selectbox("Y축 속성", selected_features, index=selected_features.index(three_defaults[1]), key="3d_y")
    with c3:
        z3 = st.selectbox("Z축 속성", selected_features, index=selected_features.index(three_defaults[2]), key="3d_z")

    fig3 = px.scatter_3d(
        work,
        x=x3,
        y=y3,
        z=z3,
        color="영화 유형",
        hover_name="영화 제목",
        hover_data={"영화 유형": True, "movieCd": True},
        category_orders={"영화 유형": CLUSTER_NAMES},
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={x3: x3, y3: y3, z3: z3},
        title=f"{x3} × {y3} × {z3}"
    )
    fig3.update_traces(marker={"size": 3, "opacity": 0.8})
    fig3.update_layout(legend_title_text="영화 유형", margin=dict(l=0, r=0, t=55, b=0))
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("마우스로 드래그해 시점을 돌리고, 휠이나 트랙패드로 확대·축소할 수 있습니다.")

st.subheader("4. 유형별 요약")
summary_rows = []
for label in CLUSTER_NAMES:
    group = work.loc[work["영화 유형"] == label]
    if group.empty:
        continue
    summary_rows.append({
        "영화 유형": label,
        "편수": len(group),
        "스크린 수 평균": group["스크린 수"].mean(),
        "누적 관객 평균": group["누적 관객"].mean(),
        "10위권 일수 평균": group["10위권 일수"].mean(),
        "롱런 지수 평균": group["롱런 지수"].mean()
    })
summary = pd.DataFrame(summary_rows)
summary["편수"] = summary["편수"].map(lambda x: f"{x:,}")
summary["스크린 수 평균"] = summary["스크린 수 평균"].map(lambda x: f"{x:,.1f}")
summary["누적 관객 평균"] = summary["누적 관객 평균"].map(lambda x: f"{x:,.0f}")
summary["10위권 일수 평균"] = summary["10위권 일수 평균"].map(lambda x: f"{x:,.1f}")
summary["롱런 지수 평균"] = summary["롱런 지수 평균"].map(lambda x: f"{x:,.2f}")
st.dataframe(summary, use_container_width=True, hide_index=True)

st.caption("유형별 평균은 원래 단위 기준입니다. 스크린 수와 관객 수는 평균 계산에 로그 변환 전 값을 사용합니다.")

st.subheader("5. 유형별 누적 관객 상위 영화")
for label in CLUSTER_NAMES:
    group = work.loc[work["영화 유형"] == label].sort_values("누적 관객", ascending=False).head(5)
    if group.empty:
        continue
    with st.expander(f"{label} 유형 · 누적 관객 상위 {len(group)}편", expanded=True):
        top = group[["movieNm", "openDt", "genre", "total_audi"]].copy()
        top.columns = ["영화 제목", "개봉일", "장르", "누적 관객"]
        top["누적 관객"] = top["누적 관객"].map(lambda x: f"{x:,.0f}")
        st.dataframe(top, use_container_width=True, hide_index=True)

with st.expander("분류 기준과 해석 안내"):
    st.markdown(
        """
        - **로그 스크린 수:** `log10(first_scrn)`
        - **로그 누적 관객:** `log10(total_audi)`
        - **10위권 일수:** `days_in_top10` 원래 값
        - **롱런 지수:** `min(total_audi / first_week_audi, 20)`
        - 네 속성 중 필요한 값이 없거나 첫 주 관객이 0 이하인 영화는 분석에서 제외합니다.
        - K-평균은 선택한 속성을 표준화한 뒤 세 군집으로 나눕니다. 난수 시드는 42로 고정했습니다.
        - ㉮·㉯·㉰는 누적 관객 평균이 높은 군집부터 붙입니다. 군집 번호 자체에는 고정된 의미가 없으므로, 유형 이름은 실행 결과의 평균에 따라 정해집니다.
        - 유형은 알고리즘이 데이터의 유사성을 기준으로 만든 묶음이며, 장르 같은 사전 정답을 학습한 결과는 아닙니다.
        """
    )
