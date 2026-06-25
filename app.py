import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Performa Akademik Siswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #151922 100%);
        border-right: 1px solid #2d3748;
    }

    /* Cards */
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

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #7c3aed22 0%, transparent 100%);
        border-left: 4px solid #7c3aed;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 24px 0 16px 0;
    }
    .section-header h3 { color: #e2e8f0; margin: 0; font-size: 1.1rem; }

    /* Upload box */
    [data-testid="stFileUploader"] {
        border: 2px dashed #7c3aed66;
        border-radius: 12px;
        padding: 12px;
        background: #1a1f2e44;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #7c3aed !important;
        border-bottom: 2px solid #7c3aed !important;
    }

    /* Dataframe */
    .dataframe { font-size: 0.85rem; }

    /* Slider label */
    .stSlider label { color: #94a3b8; }
    .stSelectbox label { color: #94a3b8; }

    /* Info box */
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

# ── Helpers ──────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    "Student_ID", "Major_Category", "Year_of_Study", "Pre_Semester_GPA",
    "Weekly_GenAI_Hours", "Primary_Use_Case", "Prompt_Engineering_Skill",
    "Tool_Diversity", "Paid_Subscription", "Traditional_Study_Hours",
]

NUMERIC_FEATURES = [
    "Year_of_Study", "Pre_Semester_GPA", "Weekly_GenAI_Hours",
    "Tool_Diversity", "Traditional_Study_Hours",
]
CATEGORICAL_FEATURES = [
    "Major_Category", "Primary_Use_Case",
    "Prompt_Engineering_Skill", "Paid_Subscription",
]

def categorize_gpa(gpa):
    """Convert numeric GPA (0–4) to performance category."""
    if gpa >= 3.5:
        return "Sangat Baik"
    elif gpa >= 3.0:
        return "Baik"
    elif gpa >= 2.5:
        return "Cukup"
    else:
        return "Kurang"

def generate_sample_data(n=200):
    """Generate realistic sample data for demonstration."""
    np.random.seed(42)
    majors = ["STEM", "Social Sciences", "Humanities", "Business", "Health Sciences"]
    use_cases = ["Research", "Writing", "Coding", "Study Aid", "Problem Solving"]
    skills   = ["Beginner", "Intermediate", "Advanced"]
    subs     = ["Yes", "No"]

    data = {
        "Student_ID":                range(1, n + 1),
        "Major_Category":            np.random.choice(majors, n),
        "Year_of_Study":             np.random.randint(1, 5, n),
        "Pre_Semester_GPA":          np.round(np.random.uniform(2.0, 4.0, n), 2),
        "Weekly_GenAI_Hours":        np.round(np.random.uniform(0, 20, n), 1),
        "Primary_Use_Case":          np.random.choice(use_cases, n),
        "Prompt_Engineering_Skill":  np.random.choice(skills, n),
        "Tool_Diversity":            np.random.randint(1, 6, n),
        "Paid_Subscription":         np.random.choice(subs, n),
        "Traditional_Study_Hours":   np.round(np.random.uniform(5, 40, n), 1),
    }
    df = pd.DataFrame(data)
    # Derive a realistic GPA that correlates with features
    df["Current_GPA"] = (
        df["Pre_Semester_GPA"] * 0.5
        + df["Weekly_GenAI_Hours"].clip(0, 10) * 0.04
        + df["Traditional_Study_Hours"] * 0.008
        + np.random.normal(0, 0.2, n)
    ).clip(0, 4).round(2)
    df["Performance_Category"] = df["Current_GPA"].apply(categorize_gpa)
    return df

def preprocess(df):
    df = df.copy()
    le = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col].astype(str))
            le[col] = enc
    return df, le

def run_knn(df, k, test_size, random_state=42):
    feature_cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df["Performance_Category"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )

    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    cv_scores = cross_val_score(knn, X_scaled, y, cv=5, scoring="accuracy")

    return {
        "model":      knn,
        "scaler":     scaler,
        "le":         {},
        "feature_cols": feature_cols,
        "X_train":    X_train, "X_test": X_test,
        "y_train":    y_train, "y_test": y_test,
        "y_pred":     y_pred,
        "accuracy":   accuracy_score(y_test, y_pred),
        "cv_scores":  cv_scores,
        "report":     classification_report(y_test, y_pred, output_dict=True),
        "cm":         confusion_matrix(y_test, y_pred, labels=knn.classes_),
        "classes":    knn.classes_,
    }

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 Pengaturan Model")
    st.markdown("---")

    st.markdown("### 📁 Upload Dataset")
    uploaded = st.file_uploader(
        "Upload file CSV",
        type=["csv"],
        help="Format kolom sesuai dataset AI Student Impact"
    )

    use_sample = st.checkbox("Gunakan data sampel (demo)", value=uploaded is None)

    st.markdown("---")
    st.markdown("### ⚙️ Hyperparameter KNN")
    k_value    = st.slider("Nilai K (tetangga)", 1, 20, 5)
    test_ratio = st.slider("Proporsi data uji (%)", 10, 40, 20) / 100

    st.markdown("---")
    st.markdown("""
    <div class='info-box'>
    <b>📌 Fitur yang digunakan:</b><br>
    • Year of Study<br>
    • Pre-Semester GPA<br>
    • Weekly GenAI Hours<br>
    • Tool Diversity<br>
    • Traditional Study Hours<br>
    • Major Category<br>
    • Primary Use Case<br>
    • Prompt Engineering Skill<br>
    • Paid Subscription
    </div>
    """, unsafe_allow_html=True)

# ── Title ────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 28px 0 18px 0;'>
    <h1 style='color:#e2e8f0; font-size:1.9rem; font-weight:700; margin:0;'>
        🎓 Prediksi Performa Akademik Siswa
    </h1>
    <p style='color:#7c3aed; font-size:1.05rem; margin:6px 0 0 0;'>
        Berdasarkan Penggunaan AI · Algoritma K-Nearest Neighbors
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────
df_raw = None
data_source = ""

if uploaded:
    try:
        df_raw = pd.read_csv(uploaded)
        df_raw.dropna(how="all", inplace=True)
        df_raw.reset_index(drop=True, inplace=True)
        data_source = f"📂 {uploaded.name}"
        st.success(f"✅ Dataset berhasil dimuat: **{len(df_raw)} baris**, **{len(df_raw.columns)} kolom**")
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")

if use_sample or df_raw is None or df_raw.empty:
    df_raw = generate_sample_data(200)
    data_source = "🔬 Data sampel (demo)"
    if not uploaded:
        st.info("ℹ️ Menggunakan **data sampel** untuk demonstrasi. Upload CSV Anda di sidebar.")

# Check target column
if "Performance_Category" not in df_raw.columns:
    if "Current_GPA" in df_raw.columns:
        df_raw["Performance_Category"] = df_raw["Current_GPA"].apply(categorize_gpa)
    elif "Pre_Semester_GPA" in df_raw.columns:
        df_raw["Performance_Category"] = df_raw["Pre_Semester_GPA"].apply(categorize_gpa)
    else:
        st.error("❌ Dataset tidak memiliki kolom GPA. Pastikan ada kolom `Current_GPA` atau `Pre_Semester_GPA`.")
        st.stop()

# Preprocess
df_proc, le_dict = preprocess(df_raw)

# ── Run model ────────────────────────────────────────────────
result = run_knn(df_proc, k=k_value, test_size=test_ratio)

# ── KPIs ─────────────────────────────────────────────────────
st.markdown("")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class='metric-card'>
        <h2>{result['accuracy']*100:.1f}%</h2>
        <p>Akurasi Model</p>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class='metric-card'>
        <h2>{result['cv_scores'].mean()*100:.1f}%</h2>
        <p>CV Accuracy (5-Fold)</p>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class='metric-card'>
        <h2>{len(df_raw)}</h2>
        <p>Total Data Siswa</p>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class='metric-card'>
        <h2>K = {k_value}</h2>
        <p>Nilai K Tetangga</p>
    </div>""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Eksplorasi Data",
    "📈 Evaluasi Model",
    "🔎 Analisis Fitur",
    "🔮 Prediksi Baru",
    "📋 Dataset",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — Eksplorasi Data
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'><h3>📊 Distribusi Performa Akademik</h3></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        perf_counts = df_raw["Performance_Category"].value_counts().reset_index()
        perf_counts.columns = ["Kategori", "Jumlah"]
        color_map = {
            "Sangat Baik": "#7c3aed",
            "Baik":        "#3b82f6",
            "Cukup":       "#f59e0b",
            "Kurang":      "#ef4444",
        }
        fig_pie = px.pie(
            perf_counts, values="Jumlah", names="Kategori",
            color="Kategori", color_discrete_map=color_map,
            title="Proporsi Kategori Performa",
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", legend_font_color="#94a3b8",
            title_font_color="#e2e8f0",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_bar = px.bar(
            perf_counts, x="Kategori", y="Jumlah",
            color="Kategori", color_discrete_map=color_map,
            title="Jumlah Siswa per Kategori",
            text="Jumlah",
        )
        fig_bar.update_traces(textposition="outside")
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", showlegend=False,
            title_font_color="#e2e8f0",
            xaxis=dict(color="#94a3b8"),
            yaxis=dict(color="#94a3b8", gridcolor="#2d3748"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # GenAI Hours vs GPA
    st.markdown("<div class='section-header'><h3>🤖 Penggunaan GenAI vs GPA</h3></div>", unsafe_allow_html=True)

    gpa_col = "Current_GPA" if "Current_GPA" in df_raw.columns else "Pre_Semester_GPA"
    if "Weekly_GenAI_Hours" in df_raw.columns:
        fig_scatter = px.scatter(
            df_raw, x="Weekly_GenAI_Hours", y=gpa_col,
            color="Performance_Category", color_discrete_map=color_map,
            title="Jam Penggunaan GenAI per Minggu vs GPA",
            labels={"Weekly_GenAI_Hours": "Jam GenAI/Minggu", gpa_col: "GPA"},
            opacity=0.75,
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#e2e8f0",
            xaxis=dict(color="#94a3b8", gridcolor="#2d3748"),
            yaxis=dict(color="#94a3b8", gridcolor="#2d3748"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Major distribution
    if "Major_Category" in df_raw.columns:
        st.markdown("<div class='section-header'><h3>🏫 Distribusi Per Jurusan</h3></div>", unsafe_allow_html=True)
        major_perf = df_raw.groupby(["Major_Category", "Performance_Category"]).size().reset_index(name="Count")
        fig_major = px.bar(
            major_perf, x="Major_Category", y="Count",
            color="Performance_Category", color_discrete_map=color_map,
            barmode="stack",
            title="Performa per Jurusan",
            labels={"Major_Category": "Jurusan", "Count": "Jumlah"},
        )
        fig_major.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#e2e8f0",
            xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8", gridcolor="#2d3748"),
        )
        st.plotly_chart(fig_major, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — Evaluasi Model
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'><h3>📈 Confusion Matrix</h3></div>", unsafe_allow_html=True)

    col_cm, col_cv = st.columns([1, 1])

    with col_cm:
        cm = result["cm"]
        classes = result["classes"]
        fig_cm = px.imshow(
            cm, x=classes, y=classes,
            color_continuous_scale="Purples",
            labels={"x": "Prediksi", "y": "Aktual", "color": "Jumlah"},
            title="Confusion Matrix",
            text_auto=True,
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#e2e8f0",
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_cv:
        cv_df = pd.DataFrame({
            "Fold": [f"Fold {i+1}" for i in range(len(result["cv_scores"]))],
            "Accuracy": result["cv_scores"] * 100,
        })
        fig_cv = px.bar(
            cv_df, x="Fold", y="Accuracy",
            color="Accuracy", color_continuous_scale="Purples",
            title="Cross-Validation Accuracy (5-Fold)",
            labels={"Accuracy": "Akurasi (%)"},
            text=cv_df["Accuracy"].round(1).astype(str) + "%",
        )
        fig_cv.add_hline(
            y=result["cv_scores"].mean() * 100,
            line_dash="dash", line_color="#f59e0b",
            annotation_text=f"Rata-rata: {result['cv_scores'].mean()*100:.1f}%",
            annotation_font_color="#f59e0b",
        )
        fig_cv.update_traces(textposition="outside")
        fig_cv.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#e2e8f0",
            yaxis=dict(range=[0, 115], gridcolor="#2d3748"),
            xaxis=dict(color="#94a3b8"),
        )
        st.plotly_chart(fig_cv, use_container_width=True)

    # Classification Report
    st.markdown("<div class='section-header'><h3>📋 Laporan Klasifikasi</h3></div>", unsafe_allow_html=True)
    report = result["report"]
    rows = []
    for label in result["classes"]:
        if label in report:
            r = report[label]
            rows.append({
                "Kategori":  label,
                "Precision": f"{r['precision']:.3f}",
                "Recall":    f"{r['recall']:.3f}",
                "F1-Score":  f"{r['f1-score']:.3f}",
                "Support":   int(r["support"]),
            })
    report_df = pd.DataFrame(rows)
    st.dataframe(report_df, use_container_width=True, hide_index=True)

    # K Optimization
    st.markdown("<div class='section-header'><h3>🔍 Optimasi Nilai K</h3></div>", unsafe_allow_html=True)
    k_range = range(1, min(21, len(df_proc) // 5))
    acc_list = []
    for k in k_range:
        knn_tmp = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
        feature_cols = result["feature_cols"]
        X_all = df_proc[feature_cols]
        scaler_tmp = StandardScaler()
        X_s = scaler_tmp.fit_transform(X_all)
        sc = cross_val_score(knn_tmp, X_s, df_proc["Performance_Category"], cv=5).mean()
        acc_list.append(sc * 100)

    fig_k = go.Figure()
    fig_k.add_trace(go.Scatter(
        x=list(k_range), y=acc_list,
        mode="lines+markers",
        line=dict(color="#7c3aed", width=2.5),
        marker=dict(size=7, color="#7c3aed"),
        name="CV Accuracy",
    ))
    fig_k.add_vline(
        x=k_value, line_dash="dash", line_color="#f59e0b",
        annotation_text=f"K terpilih = {k_value}",
        annotation_font_color="#f59e0b",
    )
    fig_k.update_layout(
        title="Akurasi KNN untuk Berbagai Nilai K",
        xaxis_title="Nilai K", yaxis_title="Akurasi (%)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", title_font_color="#e2e8f0",
        xaxis=dict(color="#94a3b8", gridcolor="#2d3748"),
        yaxis=dict(color="#94a3b8", gridcolor="#2d3748"),
    )
    st.plotly_chart(fig_k, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — Analisis Fitur
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'><h3>🔎 Distribusi Fitur Numerik</h3></div>", unsafe_allow_html=True)

    num_cols = [c for c in NUMERIC_FEATURES if c in df_raw.columns]
    color_map = {
        "Sangat Baik": "#7c3aed",
        "Baik":        "#3b82f6",
        "Cukup":       "#f59e0b",
        "Kurang":      "#ef4444",
    }

    for col in num_cols:
        fig_box = px.box(
            df_raw, x="Performance_Category", y=col,
            color="Performance_Category", color_discrete_map=color_map,
            title=f"Distribusi {col} per Kategori Performa",
            labels={"Performance_Category": "Performa", col: col.replace("_", " ")},
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#e2e8f0", showlegend=False,
            xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8", gridcolor="#2d3748"),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # Correlation heatmap
    st.markdown("<div class='section-header'><h3>🌡️ Korelasi Antar Fitur</h3></div>", unsafe_allow_html=True)
    num_df = df_proc[[c for c in NUMERIC_FEATURES if c in df_proc.columns]]
    corr = num_df.corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Matriks Korelasi Fitur Numerik",
        zmin=-1, zmax=1,
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", title_font_color="#e2e8f0",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — Prediksi Baru
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'><h3>🔮 Masukkan Data Siswa Baru</h3></div>", unsafe_allow_html=True)
    st.markdown("Isi data siswa di bawah ini untuk memprediksi kategori performa akademiknya.")

    col1, col2 = st.columns(2)

    with col1:
        year_study    = st.selectbox("Tahun Studi", [1, 2, 3, 4])
        pre_gpa       = st.slider("GPA Semester Sebelumnya", 0.0, 4.0, 3.0, 0.1)
        genai_hours   = st.slider("Jam Penggunaan GenAI/Minggu", 0.0, 20.0, 5.0, 0.5)
        trad_hours    = st.slider("Jam Belajar Tradisional/Minggu", 0.0, 40.0, 15.0, 0.5)
        tool_div      = st.slider("Keberagaman Tools AI (1–5)", 1, 5, 2)

    with col2:
        # Use actual categories from data if available
        majors_list = sorted(df_raw["Major_Category"].dropna().unique().tolist()) \
            if "Major_Category" in df_raw.columns else ["STEM", "Business", "Humanities"]
        use_cases_list = sorted(df_raw["Primary_Use_Case"].dropna().unique().tolist()) \
            if "Primary_Use_Case" in df_raw.columns else ["Research", "Coding", "Writing"]
        skills_list  = sorted(df_raw["Prompt_Engineering_Skill"].dropna().unique().tolist()) \
            if "Prompt_Engineering_Skill" in df_raw.columns else ["Beginner", "Intermediate", "Advanced"]
        subs_list    = ["Yes", "No"]

        major_cat   = st.selectbox("Jurusan", majors_list)
        use_case    = st.selectbox("Penggunaan Utama AI", use_cases_list)
        prompt_skill= st.selectbox("Keahlian Prompt Engineering", skills_list)
        paid_sub    = st.selectbox("Langganan Berbayar", subs_list)

    if st.button("🔮 Prediksi Sekarang", use_container_width=True):
        # Build input row matching df_proc encoding
        new_raw = {
            "Year_of_Study":             year_study,
            "Pre_Semester_GPA":          pre_gpa,
            "Weekly_GenAI_Hours":        genai_hours,
            "Tool_Diversity":            tool_div,
            "Traditional_Study_Hours":   trad_hours,
            "Major_Category":            major_cat,
            "Primary_Use_Case":          use_case,
            "Prompt_Engineering_Skill":  prompt_skill,
            "Paid_Subscription":         paid_sub,
        }

        # Encode categoricals using same mapping as training data
        for col in CATEGORICAL_FEATURES:
            if col in new_raw and col in df_proc.columns:
                # Map via value from df_raw → df_proc
                mapping = dict(zip(
                    df_raw[col].astype(str),
                    df_proc[col]
                ))
                val = str(new_raw[col])
                new_raw[col] = mapping.get(val, 0)

        feature_cols = result["feature_cols"]
        input_vec = np.array([[new_raw.get(c, 0) for c in feature_cols]])
        input_scaled = result["scaler"].transform(input_vec)

        prediction = result["model"].predict(input_scaled)[0]
        proba = result["model"].predict_proba(input_scaled)[0]
        classes = result["model"].classes_

        color_result = {
            "Sangat Baik": "#7c3aed",
            "Baik":        "#3b82f6",
            "Cukup":       "#f59e0b",
            "Kurang":      "#ef4444",
        }.get(prediction, "#7c3aed")

        st.markdown(f"""
        <div style='background: linear-gradient(135deg,{color_result}22,{color_result}11);
                    border: 2px solid {color_result};
                    border-radius: 14px; padding: 24px; text-align:center; margin:20px 0;'>
            <h2 style='color:{color_result}; font-size:2rem; margin:0;'>
                Prediksi: {prediction}
            </h2>
            <p style='color:#94a3b8; margin:8px 0 0 0;'>
                Hasil prediksi KNN dengan K={k_value}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Probability chart
        prob_df = pd.DataFrame({"Kategori": classes, "Probabilitas": proba * 100})
        fig_prob = px.bar(
            prob_df, x="Kategori", y="Probabilitas",
            color="Kategori",
            color_discrete_map={
                "Sangat Baik": "#7c3aed", "Baik": "#3b82f6",
                "Cukup": "#f59e0b", "Kurang": "#ef4444",
            },
            title="Probabilitas per Kategori (%)",
            text=prob_df["Probabilitas"].round(1).astype(str) + "%",
        )
        fig_prob.update_traces(textposition="outside")
        fig_prob.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", title_font_color="#e2e8f0", showlegend=False,
            xaxis=dict(color="#94a3b8"), yaxis=dict(color="#94a3b8", gridcolor="#2d3748"),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — Dataset
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown(f"<div class='section-header'><h3>📋 Dataset ({data_source})</h3></div>", unsafe_allow_html=True)

    col_s, col_e = st.columns([3, 1])
    with col_s:
        search = st.text_input("🔍 Cari dalam dataset", "")
    with col_e:
        show_n = st.selectbox("Tampilkan", [25, 50, 100, "Semua"], index=0)

    df_show = df_raw.copy()
    if search:
        mask = df_show.astype(str).apply(lambda row: row.str.contains(search, case=False)).any(axis=1)
        df_show = df_show[mask]

    if show_n != "Semua":
        df_show = df_show.head(int(show_n))

    st.dataframe(df_show, use_container_width=True, height=400)

    st.download_button(
        "⬇️ Download Dataset",
        df_raw.to_csv(index=False),
        file_name="student_performance_data.csv",
        mime="text/csv",
    )

    st.markdown("<div class='section-header'><h3>📊 Statistik Deskriptif</h3></div>", unsafe_allow_html=True)
    desc = df_raw[[c for c in NUMERIC_FEATURES if c in df_raw.columns]].describe().round(3)
    st.dataframe(desc, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#2d3748; margin-top:40px;'>
<p style='text-align:center; color:#475569; font-size:0.8rem;'>
    🎓 Prediksi Performa Akademik · Algoritma KNN · Streamlit
</p>
""", unsafe_allow_html=True)
