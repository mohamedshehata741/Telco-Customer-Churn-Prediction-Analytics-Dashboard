"""
╔══════════════════════════════════════════════════════════════════════════╗
║          🔮 Telco Customer Churn Predictor — Streamlit App               ║
║          Powered by XGBoost | Built with ❤️  & Professional UX           ║
╚══════════════════════════════════════════════════════════════════════════╝

HOW TO RUN:
    1. Place this file alongside your model:  final_telco_churn_model.pkl
    2. pip install streamlit plotly joblib xgboost scikit-learn pandas numpy
    3. streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the very first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b27; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1e2332;
        border: 1px solid #2e3550;
        border-radius: 12px;
        padding: 16px;
    }

    /* Prediction result cards */
    .result-card {
        padding: 28px 32px;
        border-radius: 16px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .card-churn {
        background: linear-gradient(135deg, #ff4b4b22 0%, #ff4b4b44 100%);
        border: 2px solid #ff4b4b;
        color: #ff4b4b;
    }
    .card-stay {
        background: linear-gradient(135deg, #00c85322 0%, #00c85344 100%);
        border: 2px solid #00c853;
        color: #00c853;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #161b27;
        border-radius: 12px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        color: #8899aa;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2e3550 !important;
        color: #ffffff !important;
    }

    /* What-if box */
    .whatif-box {
        background-color: #1e2332;
        border: 1px solid #3a4a6b;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 12px;
    }

    /* Info box */
    .info-box {
        background-color: #1a2035;
        border-left: 4px solid #4a9eff;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 0.9rem;
        color: #ccd6f6;
    }

    h1, h2, h3 { color: #e8eaf6 !important; }
    p, li, label { color: #b0bec5; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MODEL LOADING  (cached so it only loads once per session)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = "final_telco_churn_model.pkl"

@st.cache_resource(show_spinner="🔄 Loading ML pipeline…")
def load_model(path: str):
    """Load the serialised sklearn / XGBoost pipeline with robust error handling."""
    if not os.path.exists(path):
        return None, f"❌ Model file not found at `{path}`. Please place it in the same directory as app.py."
    try:
        pipeline = joblib.load(path)
        return pipeline, None
    except Exception as exc:
        return None, f"❌ Failed to load model: {exc}"

pipeline, load_error = load_model(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE SCHEMA — derived dynamically from the loaded pipeline
# ─────────────────────────────────────────────────────────────────────────────
# These are the original input columns the pipeline expects (before preprocessing).
NUMERIC_FEATURES   = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Category options extracted from the pipeline's OneHotEncoder categories
CATEGORY_OPTIONS = {
    "gender":           ["Male", "Female"],
    "Partner":          ["Yes", "No"],
    "Dependents":       ["Yes", "No"],
    "PhoneService":     ["Yes", "No"],
    "MultipleLines":    ["Yes", "No", "No phone service"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["Yes", "No", "No internet service"],
    "OnlineBackup":     ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport":      ["Yes", "No", "No internet service"],
    "StreamingTV":      ["Yes", "No", "No internet service"],
    "StreamingMovies":  ["Yes", "No", "No internet service"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod":    [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ],
}

# Typical dataset averages (Telco IBM dataset reference values)
DATASET_AVG = {
    "tenure": 32.4,
    "MonthlyCharges": 64.8,
    "TotalCharges": 2283.3,
}


# ─────────────────────────────────────────────────────────────────────────────
# 3. SIDEBAR — dynamic input form
# ─────────────────────────────────────────────────────────────────────────────
def build_sidebar() -> dict:
    """Render the sidebar input widgets and return a dict of raw feature values."""
    with st.sidebar:
        st.markdown("## 🧑‍💼 Customer Profile")
        st.markdown("---")

        st.markdown("#### 📊 Account Info")
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        contract        = st.selectbox("Contract Type", CATEGORY_OPTIONS["Contract"])
        payment_method  = st.selectbox("Payment Method", CATEGORY_OPTIONS["PaymentMethod"])
        paperless       = st.selectbox("Paperless Billing", CATEGORY_OPTIONS["PaperlessBilling"])

        st.markdown("#### 💰 Charges")
        monthly_charges = st.slider("Monthly Charges ($)", 0.0, 120.0, 65.0, step=0.5)
        # TotalCharges auto-estimated; let user override
        total_charges   = st.number_input(
            "Total Charges ($)",
            min_value=0.0, max_value=10000.0,
            value=round(float(tenure * monthly_charges), 2),
            step=10.0,
        )

        st.markdown("#### 👤 Demographics")
        gender          = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])
        senior          = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner         = st.selectbox("Partner", CATEGORY_OPTIONS["Partner"])
        dependents      = st.selectbox("Dependents", CATEGORY_OPTIONS["Dependents"])

        st.markdown("#### 📡 Services")
        phone_service   = st.selectbox("Phone Service", CATEGORY_OPTIONS["PhoneService"])
        multiple_lines  = st.selectbox("Multiple Lines", CATEGORY_OPTIONS["MultipleLines"])
        internet        = st.selectbox("Internet Service", CATEGORY_OPTIONS["InternetService"])

        st.markdown("#### 🛡️ Add-ons")
        online_security  = st.selectbox("Online Security",  CATEGORY_OPTIONS["OnlineSecurity"])
        online_backup    = st.selectbox("Online Backup",    CATEGORY_OPTIONS["OnlineBackup"])
        device_prot      = st.selectbox("Device Protection",CATEGORY_OPTIONS["DeviceProtection"])
        tech_support     = st.selectbox("Tech Support",     CATEGORY_OPTIONS["TechSupport"])
        streaming_tv     = st.selectbox("Streaming TV",     CATEGORY_OPTIONS["StreamingTV"])
        streaming_movies = st.selectbox("Streaming Movies", CATEGORY_OPTIONS["StreamingMovies"])

        st.markdown("---")
        predict_btn = st.button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

    return {
        "gender":           gender,
        "SeniorCitizen":    senior,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_prot,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
        "_predict_btn":     predict_btn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. HELPER: PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def predict(input_dict: dict):
    """
    Build a single-row DataFrame from the user inputs,
    run the pipeline, and return (label, churn_probability).
    """
    row = {k: v for k, v in input_dict.items() if k != "_predict_btn"}
    df = pd.DataFrame([row])
    # Ensure column order matches what the pipeline was trained on
    df = df[ALL_FEATURES]
    proba = pipeline.predict_proba(df)[0]  # shape (2,): [P(No), P(Yes)]
    churn_prob = float(proba[1])
    label = "Churn" if churn_prob >= 0.5 else "Stay"
    return label, churn_prob


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def gauge_chart(churn_prob: float) -> go.Figure:
    """Plotly gauge — Churn Risk Meter 0 → 100%."""
    pct = churn_prob * 100
    color = "#ff4b4b" if pct >= 50 else ("#f0a500" if pct >= 30 else "#00c853")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": "#ff4b4b"},
               "decreasing": {"color": "#00c853"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a5568",
                     "tickfont": {"color": "#8899aa"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e2332",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  30], "color": "#0d2b1d"},
                {"range": [30, 50], "color": "#2b2200"},
                {"range": [50,100], "color": "#2b0d0d"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": pct,
            },
        },
        title={"text": "Churn Risk Meter", "font": {"size": 18, "color": "#e8eaf6"}},
    ))
    fig.update_layout(
        paper_bgcolor="#0f1117", font_color="#e8eaf6",
        height=280, margin=dict(t=60, b=20, l=30, r=30),
    )
    return fig


def feature_importance_chart(pipeline) -> go.Figure:
    """Top-10 feature importances from the XGBoost booster via the full pipeline."""
    try:
        classifier = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        # Get transformed feature names
        feature_names = preprocessor.get_feature_names_out()
        importances   = classifier.feature_importances_
        fi_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(10)
            .sort_values("importance")   # ascending for horizontal bar
        )
        # Clean up prefixes (num__, cat__) for readability
        fi_df["feature"] = (
            fi_df["feature"]
            .str.replace(r"^num__", "", regex=True)
            .str.replace(r"^cat__", "", regex=True)
        )
        fig = go.Figure(go.Bar(
            x=fi_df["importance"],
            y=fi_df["feature"],
            orientation="h",
            marker=dict(
                color=fi_df["importance"],
                colorscale=[[0, "#2e3550"], [1, "#4a9eff"]],
                showscale=False,
            ),
            text=[f"{v:.3f}" for v in fi_df["importance"]],
            textposition="outside",
            textfont={"color": "#8899aa"},
        ))
        fig.update_layout(
            title=dict(text="🏆 Top 10 Feature Importances", font={"size": 16, "color": "#e8eaf6"}),
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            xaxis=dict(showgrid=True, gridcolor="#2e3550", color="#8899aa", title="Importance Score"),
            yaxis=dict(showgrid=False, color="#b0bec5"),
            height=420, margin=dict(t=50, b=40, l=20, r=60),
            font_color="#e8eaf6",
        )
        return fig
    except Exception as e:
        return None


def comparison_chart(tenure: float, monthly: float) -> go.Figure:
    """Bar chart: this customer vs dataset average for tenure & MonthlyCharges."""
    categories   = ["Tenure (months)", "Monthly Charges ($)"]
    customer_vals = [tenure, monthly]
    avg_vals      = [DATASET_AVG["tenure"], DATASET_AVG["MonthlyCharges"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="This Customer", x=categories, y=customer_vals,
        marker_color="#4a9eff",
        text=[f"{v:.1f}" for v in customer_vals],
        textposition="auto",
    ))
    fig.add_trace(go.Bar(
        name="Dataset Average", x=categories, y=avg_vals,
        marker_color="#8c54ff",
        text=[f"{v:.1f}" for v in avg_vals],
        textposition="auto",
    ))
    fig.update_layout(
        title=dict(text="📊 Customer vs. Dataset Average", font={"size": 16, "color": "#e8eaf6"}),
        barmode="group",
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        xaxis=dict(showgrid=False, color="#8899aa"),
        yaxis=dict(showgrid=True, gridcolor="#2e3550", color="#8899aa"),
        legend=dict(bgcolor="#1e2332", bordercolor="#2e3550", borderwidth=1),
        height=360, margin=dict(t=50, b=40, l=20, r=20),
        font_color="#e8eaf6",
    )
    return fig


def whatif_chart(base_input: dict, current_contract: str) -> go.Figure:
    """What-If Analysis: compare churn risk across all three contract types."""
    contracts = ["Month-to-month", "One year", "Two year"]
    risks     = []
    for c in contracts:
        modified = {**base_input, "Contract": c}
        _, prob  = predict(modified)
        risks.append(prob * 100)

    colors = ["#ff4b4b" if r >= 50 else "#f0a500" if r >= 30 else "#00c853" for r in risks]
    fig = go.Figure(go.Bar(
        x=contracts, y=risks,
        marker_color=colors,
        text=[f"{r:.1f}%" for r in risks],
        textposition="auto",
        textfont={"size": 14, "color": "white"},
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#8899aa",
                  annotation_text="50% threshold", annotation_font_color="#8899aa")
    fig.update_layout(
        title=dict(text="🔄 What-If: Churn Risk by Contract Type", font={"size": 16, "color": "#e8eaf6"}),
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        xaxis=dict(showgrid=False, color="#8899aa"),
        yaxis=dict(showgrid=True, gridcolor="#2e3550", color="#8899aa",
                   title="Churn Risk (%)", range=[0, 105]),
        height=360, margin=dict(t=50, b=40, l=20, r=20),
        font_color="#e8eaf6",
    )
    # Highlight the current contract
    current_idx = contracts.index(current_contract)
    fig.add_annotation(
        x=current_contract, y=risks[current_idx] + 5,
        text="← current",
        showarrow=False,
        font=dict(color="#ffdd57", size=12),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN APP LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── App Header ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style='text-align:center; color:#e8eaf6; margin-bottom:0'>
            🔮 Telco Customer Churn Predictor
        </h1>
        <p style='text-align:center; color:#607d8b; font-size:1rem; margin-top:4px'>
            XGBoost · Scikit-learn Pipeline · Real-time Risk Scoring
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── Model load error guard ───────────────────────────────────────────────
    if load_error:
        st.error(load_error)
        st.info("💡 Make sure `final_telco_churn_model.pkl` is in the same folder as `app.py`.")
        st.stop()

    # ── Collect sidebar inputs ───────────────────────────────────────────────
    inputs = build_sidebar()
    predict_btn = inputs.pop("_predict_btn")

    # Session state: store last prediction so UI persists across widget changes
    if "last_label"      not in st.session_state: st.session_state["last_label"]      = None
    if "last_prob"       not in st.session_state: st.session_state["last_prob"]        = None
    if "last_inputs"     not in st.session_state: st.session_state["last_inputs"]      = None

    if predict_btn:
        try:
            label, prob = predict(inputs)
            st.session_state["last_label"]  = label
            st.session_state["last_prob"]   = prob
            st.session_state["last_inputs"] = inputs.copy()
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Customer Analytics", "🧠 Model Info"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — PREDICTION
    # ════════════════════════════════════════════════════════════════════════
    with tab1:
        if st.session_state["last_label"] is None:
            st.markdown(
                """
                <div class='info-box'>
                    👈 Fill in the <strong>Customer Profile</strong> on the left sidebar,
                    then hit <strong>🔮 Predict Churn Risk</strong> to get started.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            label = st.session_state["last_label"]
            prob  = st.session_state["last_prob"]
            pct   = prob * 100

            # ── Result card ─────────────────────────────────────────────────
            card_class = "card-churn" if label == "Churn" else "card-stay"
            icon       = "🔴" if label == "Churn" else "🟢"
            headline   = "⚠️ HIGH CHURN RISK" if label == "Churn" else "✅ LIKELY TO STAY"
            sub        = f"Churn probability: {pct:.1f}%"

            st.markdown(
                f"""
                <div class='result-card {card_class}'>
                    {icon} {headline}<br>
                    <span style='font-size:1.1rem; font-weight:400; opacity:0.85'>{sub}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Gauge + key metrics ─────────────────────────────────────────
            col_gauge, col_metrics = st.columns([1.3, 1])

            with col_gauge:
                st.plotly_chart(gauge_chart(prob), use_container_width=True)

            with col_metrics:
                st.markdown("### 📋 Key Inputs")
                last = st.session_state["last_inputs"]
                st.metric("Tenure",           f"{last['tenure']} months")
                st.metric("Monthly Charges",  f"${last['MonthlyCharges']:.2f}")
                st.metric("Contract",          last["Contract"])
                st.metric("Internet Service",  last["InternetService"])

            st.divider()

            # ── What-If Analysis ────────────────────────────────────────────
            st.markdown("### 🔄 What-If Analysis")
            st.markdown(
                "<div class='info-box'>See how changing the <strong>Contract Type</strong> "
                "would affect this customer's churn risk — all other inputs remain the same.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div class='whatif-box'>", unsafe_allow_html=True)
            wi_fig = whatif_chart(
                st.session_state["last_inputs"],
                st.session_state["last_inputs"]["Contract"],
            )
            st.plotly_chart(wi_fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — CUSTOMER ANALYTICS
    # ════════════════════════════════════════════════════════════════════════
    with tab2:
        if st.session_state["last_inputs"] is None:
            st.markdown(
                "<div class='info-box'>Run a prediction first to see customer analytics.</div>",
                unsafe_allow_html=True,
            )
        else:
            last = st.session_state["last_inputs"]

            # ── Feature Importance ──────────────────────────────────────────
            st.markdown("### 🏆 Feature Importance")
            fi_fig = feature_importance_chart(pipeline)
            if fi_fig:
                st.plotly_chart(fi_fig, use_container_width=True)
            else:
                st.warning("Feature importances unavailable for this model variant.")

            st.divider()

            # ── Customer vs Average ─────────────────────────────────────────
            st.markdown("### 📊 Customer vs. Dataset Average")
            cmp_fig = comparison_chart(last["tenure"], last["MonthlyCharges"])
            st.plotly_chart(cmp_fig, use_container_width=True)

            st.markdown(
                f"""
                <div class='info-box'>
                    📌 This customer's <b>tenure</b> is
                    <b>{last['tenure']}</b> months vs. dataset avg of <b>{DATASET_AVG['tenure']}</b> months.<br>
                    Their <b>monthly charges</b> are
                    <b>${last['MonthlyCharges']:.2f}</b> vs. avg of <b>${DATASET_AVG['MonthlyCharges']:.2f}</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — MODEL INFO
    # ════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("### 🧠 Pipeline Architecture")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### ⚙️ Pipeline Steps")
            for step_name, step_obj in pipeline.steps:
                st.markdown(
                    f"- **`{step_name}`** → `{type(step_obj).__name__}`"
                )

            st.markdown("#### 🔢 Numeric Features")
            for f in NUMERIC_FEATURES:
                st.markdown(f"  - `{f}`")

            st.markdown("#### 🔤 Categorical Features")
            for f in CATEGORICAL_FEATURES:
                st.markdown(f"  - `{f}`")

        with col_b:
            st.markdown("#### 🌳 Classifier Hyperparameters")
            try:
                clf = pipeline.named_steps["classifier"]
                params = clf.get_params()
                key_params = {
                    "n_estimators": params.get("n_estimators"),
                    "max_depth":    params.get("max_depth"),
                    "learning_rate":params.get("learning_rate"),
                    "objective":    params.get("objective"),
                    "scale_pos_weight": params.get("scale_pos_weight"),
                    "random_state": params.get("random_state"),
                }
                for k, v in key_params.items():
                    if v is not None:
                        st.markdown(f"- **{k}**: `{v}`")
            except Exception:
                st.info("Hyperparameter details not available.")

        st.divider()
        st.markdown(
            """
            <div class='info-box'>
            <b>📌 About this model</b><br>
            This pipeline was trained on the <em>IBM Telco Customer Churn</em> dataset.
            It combines a <code>ColumnTransformer</code> (median imputation + standard scaling
            for numerics; one-hot encoding for categoricals) with an <code>XGBClassifier</code>
            optimised for binary churn detection.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()