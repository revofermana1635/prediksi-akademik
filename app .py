import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Prediksi Performa Akademik Pelajar Berdasarkan Penggunaan AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #151922 100%);
        border-right: 1px solid #2d3748;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2535 0%, #252d3d 100%);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-card h2 { color: #7c3aed; font-size: 2.2rem; margin: 0; }
    .metric-card p  { color: #94a3b8; margin: 4px 0 0 0; font-size: 0.9rem; }
    .section-header {
        background: linear-gradient(90deg, #7c3aed22 0%, transparent 100%);
        border-left: 4px solid #7c3aed;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
    }
    .section-header h3 { color: #e2e8f0; margin: 0; font-size: 1.1rem; }
    .info-box {
        background: #1e2535;
        border: 1px solid #7c3aed44;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Kolom kategorikal dan numerik dari dataset asli
CATEGORICAL_COLS = [
    "Major_Category", "Year_of_Study", "Primary_Use_Case",
    "Prompt_Engineering_Skill", "Institutional_Policy", "Burnout_Risk_Level"
]
BOOL_COLS = ["Paid_Subscription"]
NUMERIC_COLS = [
    "Pre_Semester_GPA", "Weekly_GenAI_Hours", "Tool_Diversity",
    "Traditional_Study_Hours", "Perceived_AI_Dependency",
    "Anxiety_Level_During_Exams", "Skill_Retention_Score"
]
TARGET_COL = "Post_Semester_GPA"

COLOR_MAP = {
    "Sangat Baik": "#7c3aed",
    "Baik":        "#3b82f6",
    "Cukup":       "#f59e0b",
    "Kurang":      "#ef4444",
}

def categorize_gpa(gpa):
    if gpa >= 3.5:   return "Sangat Baik"
    elif gpa >= 3.0: return "Baik"
    elif gpa >= 2.5: return "Cukup"
    else:            return "Kurang"

def preprocess_df(df):
    df = df.copy()
    # Encode boolean
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).map({"True": 1, "False": 0, "true": 1, "false": 0, "1": 1, "0": 0}).fillna(0).astype(int)
    # Encode categoricals
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

def get_feature_cols(df):
    available = [c for c in NUMERIC_COLS + CATEGORICAL_COLS + BOOL_COLS if c in df.columns]
    return available

def run_knn(df_proc, k, test_size):
    feature_cols = get_feature_cols(df_proc)
    X = df_proc[feature_cols].values.astype(float)
    y = df_proc["Performance_Category"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    cv_scores = cross_val_score(knn, X_scaled, y, cv=5, scoring="accuracy")

    return {
        "model": knn, "scaler": scaler,
        "feature_cols": feature_cols,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "cv_scores": cv_scores,
        "report": classification_report(y_test, y_pred, output_dict=True),
        "cm": confusion_matrix(y_test, y_pred, labels=knn.classes_),
        "classes": knn.classes_,
    }

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Pengaturan Model")
    st.markdown("---")
    st.markdown("### Upload Dataset")
    uploaded = st.file_uploader("Upload file CSV", type=["csv"])
    st.markdown("---")
    st.markdown("### Hyperparameter KNN")
    k_value    = st.slider("Nilai K (tetangga)", 1, 20, 5)
    test_ratio = st.slider("Proporsi data uji (%)", 10, 40, 20) / 100
    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b> Fitur yang digunakan:</b><br>
    • Pre-Semester GPA<br>
    • Weekly GenAI Hours<br>
    • Tool Diversity<br>
    • Traditional Study Hours<br>
    • Perceived AI Dependency<br>
    • Anxiety Level During Exams<br>
    • Skill Retention Score<br>
    • Major Category<br>
    • Year of Study<br>
    • Primary Use Case<br>
    • Prompt Engineering Skill<br>
    • Institutional Policy<br>
    • Paid Subscription<br>
    • Burnout Risk Level
    </div>
    """, unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 28px 0 18px 0;'>
    <h1 style='color:#e2e8f0; font-size:1.9rem; font-weight:700; margin:0;'>
            page_title="Prediksi Performa Akademik Pelajar Berdasarkan Penggunaan AI",
    </h1>
    <p style='color:#7c3aed; font-size:1.05rem; margin:6px 0 0 0;'>
        Menggunakan Algoritma K-Nearest Neighbors (K-NN)
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────
df_raw = None

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
        df_raw.dropna(how="all", inplace=True)
        df_raw.reset_index(drop=True, inplace=True)
        st.success(f"Dataset berhasil dimuat: **{len(df_raw):,} baris**, **{len(df_raw.columns)} kolom**")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        df_raw = None

if df_raw is None or df_raw.empty:
    st.info("Upload dataset CSV Anda di sidebar untuk memulai analisis.")
    st.stop()

# Buat kolom target
if "Post_Semester_GPA" in df_raw.columns:
    df_raw["Performance_Category"] = df_raw["Post_Semester_GPA"].apply(categorize_gpa)
elif "Pre_Semester_GPA" in df_raw.columns:
    df_raw["Performance_Category"] = df_raw["Pre_Semester_GPA"].apply(categorize_gpa)
else:
    st.error("Tidak ditemukan kolom GPA.")
    st.stop()

# Sample jika terlalu besar (untuk kecepatan)
df_use = df_raw.copy()
if len(df_use) > 5000:
    df_use = df_use.sample(5000, random_state=42).reset_index(drop=True)
    st.warning(f"⚡ Dataset besar ({len(df_raw):,} baris). Menggunakan sampel **5.000 baris** untuk kecepatan.")

df_proc, encoders = preprocess_df(df_use)

# ── Run KNN ──────────────────────────────────────────────────
try:
    result = run_knn(df_proc, k=k_value, test_size=test_ratio)
except Exception as e:
    st.error(f"Error saat menjalankan KNN: {e}")
    st.stop()

# ── KPIs ─────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class='metric-card'><h2>{result['accuracy']*100:.1f}%</h2><p>Akurasi Model</p></div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class='metric-card'><h2>{result['cv_scores'].mean()*100:.1f}%</h2><p>CV Accuracy (5-Fold)</p></div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class='metric-card'><h2>{len(df_raw):,}</h2><p>Total Data Siswa</p></div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class='metric-card'><h2>K = {k_value}</h2><p>Nilai K Tetangga</p></div>""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Eksplorasi Data", "Evaluasi Model",
    "Analisis Fitur", "Prediksi Baru", "Dataset"
])

# ── TAB 1: Eksplorasi Data ───────────────────────────────────
with tab1:
    st.markdown("<div class='section-header'><h3>Distribusi Performa Akademik</h3></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    perf_counts = df_raw["Performance_Category"].value_counts().reset_index()
    perf_counts.columns = ["Kategori", "Jumlah"]

    with col_a:
        fig = px.pie(perf_counts, values="Jumlah", names="Kategori",
                     color="Kategori", color_discrete_map=COLOR_MAP,
                     title="Proporsi Kategori Performa", hole=0.45)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.bar(perf_counts, x="Kategori", y="Jumlah",
                     color="Kategori", color_discrete_map=COLOR_MAP,
                     title="Jumlah Siswa per Kategori", text="Jumlah")
        fig.update_traces(textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", showlegend=False,
                          yaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(fig, use_container_width=True)

    if "Weekly_GenAI_Hours" in df_raw.columns and "Post_Semester_GPA" in df_raw.columns:
        st.markdown("<div class='section-header'><h3>Penggunaan GenAI vs GPA</h3></div>", unsafe_allow_html=True)
        sample = df_raw.sample(min(2000, len(df_raw)), random_state=42)
        fig = px.scatter(sample, x="Weekly_GenAI_Hours", y="Post_Semester_GPA",
                         color="Performance_Category", color_discrete_map=COLOR_MAP,
                         title="Jam GenAI per Minggu vs Post-Semester GPA", opacity=0.6)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", xaxis=dict(gridcolor="#2d3748"),
                          yaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(fig, use_container_width=True)

    if "Major_Category" in df_raw.columns:
        st.markdown("<div class='section-header'><h3>Distribusi Per Jurusan</h3></div>", unsafe_allow_html=True)
        major_perf = df_raw.groupby(["Major_Category", "Performance_Category"]).size().reset_index(name="Count")
        fig = px.bar(major_perf, x="Major_Category", y="Count",
                     color="Performance_Category", color_discrete_map=COLOR_MAP,
                     barmode="stack", title="Performa per Jurusan")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", yaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(fig, use_container_width=True)

    if "Year_of_Study" in df_raw.columns:
        st.markdown("<div class='section-header'><h3>Distribusi Per Tahun Studi</h3></div>", unsafe_allow_html=True)
        year_perf = df_raw.groupby(["Year_of_Study", "Performance_Category"]).size().reset_index(name="Count")
        fig = px.bar(year_perf, x="Year_of_Study", y="Count",
                     color="Performance_Category", color_discrete_map=COLOR_MAP,
                     barmode="group", title="Performa per Tahun Studi")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", yaxis=dict(gridcolor="#2d3748"))
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 2: Evaluasi Model ────────────────────────────────────
with tab2:
    st.markdown("<div class='section-header'><h3>Confusion Matrix & Cross-Validation</h3></div>", unsafe_allow_html=True)
    col_cm, col_cv = st.columns(2)

    with col_cm:
        fig = px.imshow(result["cm"], x=result["classes"], y=result["classes"],
                        color_continuous_scale="Purples", text_auto=True,
                        labels={"x": "Prediksi", "y": "Aktual"},
                        title="Confusion Matrix")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with col_cv:
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(5)],
            "Accuracy": result["cv_scores"] * 100
        })
        fig = px.bar(cv_df, x="Fold", y="Accuracy", color="Accuracy",
                     color_continuous_scale="Purples",
                     title="Cross-Validation Accuracy (5-Fold)",
                     text=cv_df["Accuracy"].round(1).astype(str) + "%")
        fig.add_hline(y=result["cv_scores"].mean()*100, line_dash="dash",
                      line_color="#f59e0b",
                      annotation_text=f"Rata-rata: {result['cv_scores'].mean()*100:.1f}%",
                      annotation_font_color="#f59e0b")
        fig.update_traces(textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", yaxis=dict(range=[0, 115], gridcolor="#2d3748"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'><h3>Laporan Klasifikasi</h3></div>", unsafe_allow_html=True)
    rows = []
    for label in result["classes"]:
        if label in result["report"]:
            r = result["report"][label]
            rows.append({"Kategori": label, "Precision": f"{r['precision']:.3f}",
                         "Recall": f"{r['recall']:.3f}", "F1-Score": f"{r['f1-score']:.3f}",
                         "Support": int(r["support"])})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-header'><h3>Optimasi Nilai K</h3></div>", unsafe_allow_html=True)
    k_range = range(1, 21)
    acc_list = []
    feature_cols = result["feature_cols"]
    X_all = df_proc[feature_cols].values.astype(float)
    scaler_tmp = StandardScaler()
    X_s = scaler_tmp.fit_transform(X_all)
    for k in k_range:
        sc = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_s,
                             df_proc["Performance_Category"], cv=5).mean()
        acc_list.append(sc * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(k_range), y=acc_list, mode="lines+markers",
                             line=dict(color="#7c3aed", width=2.5),
                             marker=dict(size=7, color="#7c3aed")))
    fig.add_vline(x=k_value, line_dash="dash", line_color="#f59e0b",
                  annotation_text=f"K={k_value}", annotation_font_color="#f59e0b")
    fig.update_layout(title="Akurasi KNN untuk Berbagai Nilai K",
                      xaxis_title="Nilai K", yaxis_title="Akurasi (%)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e2e8f0", xaxis=dict(gridcolor="#2d3748"),
                      yaxis=dict(gridcolor="#2d3748"))
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: Analisis Fitur ────────────────────────────────────
with tab3:
    st.markdown("<div class='section-header'><h3>Distribusi Fitur Numerik per Kategori</h3></div>", unsafe_allow_html=True)
    num_available = [c for c in NUMERIC_COLS if c in df_raw.columns]
    selected_feat = st.selectbox("Pilih fitur:", num_available)
    fig = px.box(df_raw, x="Performance_Category", y=selected_feat,
                 color="Performance_Category", color_discrete_map=COLOR_MAP,
                 title=f"Distribusi {selected_feat} per Kategori Performa")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e2e8f0", showlegend=False,
                      yaxis=dict(gridcolor="#2d3748"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-header'><h3>Korelasi Antar Fitur Numerik</h3></div>", unsafe_allow_html=True)
    corr_df = df_proc[[c for c in NUMERIC_COLS if c in df_proc.columns]].corr()
    fig = px.imshow(corr_df, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Matriks Korelasi", zmin=-1, zmax=1)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    st.plotly_chart(fig, use_container_width=True)

# ── TAB 4: Prediksi Baru ─────────────────────────────────────
with tab4:
    st.markdown("<div class='section-header'><h3>Masukkan Data Siswa Baru</h3></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        pre_gpa      = st.slider("Pre-Semester GPA", 0.0, 4.0, 3.0, 0.1)
        genai_hours  = st.slider("Jam GenAI/Minggu", 0.0, 30.0, 5.0, 0.5)
        trad_hours   = st.slider("Jam Belajar Tradisional/Minggu", 0.0, 40.0, 15.0, 0.5)
        tool_div     = st.slider("Tool Diversity (1-5)", 1, 5, 2)
        ai_dep       = st.slider("Perceived AI Dependency (1-10)", 1, 10, 5)
        anxiety      = st.slider("Anxiety Level During Exams (1-10)", 1, 10, 5)
        skill_ret    = st.slider("Skill Retention Score", 0.0, 100.0, 70.0, 1.0)

    with col2:
        major_opts   = sorted(df_raw["Major_Category"].dropna().unique()) if "Major_Category" in df_raw.columns else ["STEM"]
        year_opts    = sorted(df_raw["Year_of_Study"].dropna().unique()) if "Year_of_Study" in df_raw.columns else ["Freshman"]
        usecase_opts = sorted(df_raw["Primary_Use_Case"].dropna().unique()) if "Primary_Use_Case" in df_raw.columns else ["Research"]
        skill_opts   = sorted(df_raw["Prompt_Engineering_Skill"].dropna().unique()) if "Prompt_Engineering_Skill" in df_raw.columns else ["Beginner"]
        policy_opts  = sorted(df_raw["Institutional_Policy"].dropna().unique()) if "Institutional_Policy" in df_raw.columns else ["Allowed_With_Citation"]
        burnout_opts = sorted(df_raw["Burnout_Risk_Level"].dropna().unique()) if "Burnout_Risk_Level" in df_raw.columns else ["Low"]

        major      = st.selectbox("Jurusan", major_opts)
        year       = st.selectbox("Tahun Studi", year_opts)
        use_case   = st.selectbox("Primary Use Case AI", usecase_opts)
        prompt_sk  = st.selectbox("Prompt Engineering Skill", skill_opts)
        policy     = st.selectbox("Institutional Policy", policy_opts)
        burnout    = st.selectbox("Burnout Risk Level", burnout_opts)
        paid_sub   = st.selectbox("Paid Subscription", ["True", "False"])

    if st.button("Prediksi Sekarang", use_container_width=True):
        new_data = {
            "Pre_Semester_GPA": pre_gpa,
            "Weekly_GenAI_Hours": genai_hours,
            "Tool_Diversity": tool_div,
            "Traditional_Study_Hours": trad_hours,
            "Perceived_AI_Dependency": ai_dep,
            "Anxiety_Level_During_Exams": anxiety,
            "Skill_Retention_Score": skill_ret,
            "Major_Category": major,
            "Year_of_Study": year,
            "Primary_Use_Case": use_case,
            "Prompt_Engineering_Skill": prompt_sk,
            "Institutional_Policy": policy,
            "Burnout_Risk_Level": burnout,
            "Paid_Subscription": 1 if paid_sub == "True" else 0,
        }

        # Encode categorical using same encoder
        for col, enc in encoders.items():
            if col in new_data:
                val = str(new_data[col])
                if val in enc.classes_:
                    new_data[col] = enc.transform([val])[0]
                else:
                    new_data[col] = 0

        feature_cols = result["feature_cols"]
        input_vec = np.array([[new_data.get(c, 0) for c in feature_cols]], dtype=float)
        input_scaled = result["scaler"].transform(input_vec)

        prediction = result["model"].predict(input_scaled)[0]
        proba = result["model"].predict_proba(input_scaled)[0]
        classes = result["model"].classes_

        color_r = COLOR_MAP.get(prediction, "#7c3aed")
        st.markdown(f"""
        <div style='background:{color_r}22; border:2px solid {color_r};
                    border-radius:14px; padding:24px; text-align:center; margin:20px 0;'>
            <h2 style='color:{color_r}; font-size:2rem; margin:0;'>Prediksi: {prediction}</h2>
            <p style='color:#94a3b8; margin:8px 0 0 0;'>Hasil prediksi KNN dengan K={k_value}</p>
        </div>
        """, unsafe_allow_html=True)

        prob_df = pd.DataFrame({"Kategori": classes, "Probabilitas": proba * 100})
        fig = px.bar(prob_df, x="Kategori", y="Probabilitas",
                     color="Kategori", color_discrete_map=COLOR_MAP,
                     title="Probabilitas per Kategori (%)",
                     text=prob_df["Probabilitas"].round(1).astype(str) + "%")
        fig.update_traces(textposition="outside")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", showlegend=False,
                          yaxis=dict(range=[0, 115], gridcolor="#2d3748"))
        st.plotly_chart(fig, use_container_width=True)

# ── TAB 5: Dataset ───────────────────────────────────────────
with tab5:
    st.markdown(f"<div class='section-header'><h3>Dataset ({len(df_raw):,} baris)</h3></div>", unsafe_allow_html=True)
    search = st.text_input("Cari dalam dataset", "")
    show_n = st.selectbox("Tampilkan", [25, 50, 100, "Semua"])
    df_show = df_raw.copy()
    if search:
        mask = df_show.astype(str).apply(lambda row: row.str.contains(search, case=False)).any(axis=1)
        df_show = df_show[mask]
    if show_n != "Semua":
        df_show = df_show.head(int(show_n))
    st.dataframe(df_show, use_container_width=True, height=400)
    st.download_button("⬇️ Download Dataset", df_raw.to_csv(index=False),
                       file_name="student_data.csv", mime="text/csv")
    st.markdown("<div class='section-header'><h3>Statistik Deskriptif</h3></div>", unsafe_allow_html=True)
    st.dataframe(df_raw[[c for c in NUMERIC_COLS if c in df_raw.columns]].describe().round(3),
                 use_container_width=True)

st.markdown("""
<hr style='border-color:#2d3748; margin-top:40px;'>
<p style='text-align:center; color:#475569; font-size:0.8rem;'>
    🎓 Prediksi Performa Akademik · Algoritma K-NN · Streamlit
</p>
""", unsafe_allow_html=True)
