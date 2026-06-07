# app1.py

import os
from textwrap import dedent

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# Page configuration
# ============================================================

st.set_page_config(
    page_title="Medical insurance cost prediction",
    page_icon="💼",
    layout="wide"
)


# ============================================================
# File paths
# ============================================================

MODEL_PATH = "artifacts/final_medical_insurance_model.pkl"
METADATA_PATH = "artifacts/model_metadata.pkl"
DATA_PATH = "Health_Insurance.csv"
BANNER_PATH = "assets/med-insurance-banner.png"


# ============================================================
# Styling
# ============================================================

st.markdown(
    dedent("""
    <style>
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        .section-card {
            background-color: #f7f7f8;
            border-left: 5px solid #e00022;
            padding: 24px 26px;
            border-radius: 14px;
            margin-top: 18px;
            margin-bottom: 28px;
        }

        .section-card h2 {
            margin-bottom: 10px;
            color: #2b2d35;
        }

        .section-card p {
            color: #30323a;
            font-size: 0.98rem;
            line-height: 1.65;
        }

        .input-summary-card {
            background-color: #f7f7f8;
            border: 1px solid #ededed;
            border-radius: 14px;
            padding: 18px 20px;
            min-height: 112px;
            margin-bottom: 10px;
        }

        .input-summary-title {
            font-size: 0.82rem;
            color: #666666;
            font-weight: 700;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.03em;
        }

        .input-summary-value {
            font-size: 1.08rem;
            color: #2b2d35;
            font-weight: 800;
            line-height: 1.35;
        }

        .input-summary-note {
            font-size: 0.82rem;
            color: #6b6b6b;
            margin-top: 6px;
        }

        .metric-card {
            background-color: #f7f7f8;
            border: 1px solid #ededed;
            border-radius: 14px;
            padding: 20px 22px;
            min-height: 122px;
            text-align: center;
        }

        .metric-title {
            font-size: 0.84rem;
            color: #666666;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .metric-value {
            font-size: 1.85rem;
            line-height: 1.15;
            font-weight: 850;
            color: #2b2d35;
            letter-spacing: -0.4px;
        }

        .metric-value-green {
            color: #137a35;
        }

        .metric-value-neutral {
            color: #2b2d35;
            font-size: 1.35rem;
        }

        .insight-card {
            background-color: #f7f7f8;
            border-left: 5px solid #e00022;
            padding: 22px 24px;
            border-radius: 14px;
            margin-top: 20px;
            margin-bottom: 28px;
        }

        .insight-card h3 {
            margin-bottom: 12px;
            color: #2b2d35;
        }

        .insight-card p {
            color: #2b2d35;
            line-height: 1.6;
            font-size: 0.98rem;
        }

        .dashboard-question {
            font-size: 1.35rem;
            font-weight: 800;
            color: #2b2d35;
            margin-top: 34px;
            margin-bottom: 12px;
        }

        .footer-note {
            font-size: 0.82rem;
            color: #6b6b6b;
            border-top: 1px solid #dddddd;
            padding-top: 14px;
            margin-top: 34px;
            line-height: 1.5;
        }

        div.stButton > button[kind="primary"] {
            background-color: #8b0000;
            color: white;
            border: 1px solid #8b0000;
            border-radius: 10px;
            padding: 0.68rem 1.25rem;
            font-weight: 800;
            font-size: 0.95rem;
            box-shadow: 0 4px 12px rgba(139, 0, 0, 0.18);
        }

        div.stButton > button[kind="primary"]:hover {
            background-color: #a60000;
            color: white;
            border: 1px solid #a60000;
        }

        div.stButton > button[kind="primary"]:active {
            background-color: #6f0000;
            color: white;
            border: 1px solid #6f0000;
        }
    </style>
    """),
    unsafe_allow_html=True
)


# ============================================================
# Feature engineering helpers
# ============================================================

def create_age_group(age):
    if age <= 25:
        return "18-25"
    elif age <= 35:
        return "26-35"
    elif age <= 45:
        return "36-45"
    elif age <= 55:
        return "46-55"
    else:
        return "56-64"


def create_bmi_category(bmi):
    if bmi <= 18.5:
        return "Underweight"
    elif bmi <= 25:
        return "Normal"
    elif bmi <= 30:
        return "Overweight"
    elif bmi <= 35:
        return "Obese I"
    else:
        return "Obese II+"


def add_engineered_features(data):
    data = data.copy()

    data["age_group"] = pd.cut(
        data["age"],
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-64"]
    ).astype(str)

    data["bmi_category"] = pd.cut(
        data["bmi"],
        bins=[0, 18.5, 25, 30, 35, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese I", "Obese II+"]
    ).astype(str)

    data["smoker_bmi_interaction"] = (
        data["smoker"].astype(str) + "_" + data["bmi_category"].astype(str)
    )

    return data


# ============================================================
# Load model, metadata, and data
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_metadata():
    return joblib.load(METADATA_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data = data.drop_duplicates().reset_index(drop=True)
    return add_engineered_features(data)


try:
    model = load_model()
    metadata = load_metadata()
    df = load_data()
except Exception as error:
    st.error("The app could not load the required model, metadata, or dataset files.")
    st.write(error)
    st.stop()


# ============================================================
# General helper functions
# ============================================================

def format_currency(value):
    return f"${value:,.0f}"


def get_cost_segment(predicted_charge, q25, q75, q90):
    if predicted_charge <= q25:
        return "Low cost segment"
    elif predicted_charge <= q75:
        return "Medium cost segment"
    elif predicted_charge <= q90:
        return "High cost segment"
    else:
        return "Very high cost segment"


def get_profile_summary(smoker, bmi_category, age_group):
    if smoker == "yes" and bmi_category in ["Obese I", "Obese II+"]:
        return "High cost-driver combination"
    if smoker == "yes":
        return "Smoking-led cost-driver segment"
    if bmi_category in ["Obese I", "Obese II+"]:
        return "BMI-led cost-driver segment"
    if age_group in ["46-55", "56-64"]:
        return "Age-led cost-driver segment"
    return "Standard cost-driver segment"


def get_recommendation_for_prediction(smoker, bmi_category, age_group):
    if smoker == "yes" and bmi_category in ["Obese I", "Obese II+"]:
        return (
            "This profile combines smoking status with a higher BMI category. In the dataset, this combination "
            "is associated with higher insurance charges. Management use case: risk monitoring, reserve planning, "
            "and wellness-program targeting."
        )

    if smoker == "yes":
        return (
            "Smoking status is one of the strongest cost drivers in the dataset. Management use case: pricing review, "
            "risk monitoring, and preventive health engagement."
        )

    if bmi_category in ["Obese I", "Obese II+"]:
        return (
            "BMI category may contribute to higher expected costs, although the effect is much stronger when combined "
            "with smoking. Management use case: wellness-program focus."
        )

    if age_group in ["46-55", "56-64"]:
        return (
            "Older age groups show higher average insurance charges in the dataset. Management use case: reserve "
            "planning and customer segment monitoring."
        )

    return (
        "This profile does not fall into the highest visible cost-driver combinations from the dataset. The prediction "
        "should be interpreted as an educational ML estimate, not as an underwriting or pricing rule."
    )


def get_range_note(rmse):
    return (
        f"The estimated range uses the final model RMSE of about {format_currency(rmse)}. "
        "Actual charges may vary around the predicted value."
    )


def make_segment_risk_level(avg_charge, q75, q90):
    if avg_charge >= q90:
        return "Very high"
    elif avg_charge >= q75:
        return "High"
    else:
        return "Moderate"


def make_segment_action(risk_level, smoker_value, bmi_category):
    if risk_level == "Very high":
        return "Reserve planning + pricing review + wellness focus"
    if smoker_value == "yes" and bmi_category in ["Obese I", "Obese II+"]:
        return "Risk monitoring + wellness intervention"
    if risk_level == "High":
        return "Pricing review + segment monitoring"
    return "Monitor trend"


def clean_feature_name(feature_name):
    cleaned = feature_name
    cleaned = cleaned.replace("num__", "")
    cleaned = cleaned.replace("cat__", "")
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.replace("smoker bmi interaction", "smoker x bmi")
    return cleaned.title()


def get_feature_importance_from_linear_pipeline(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    model_step = pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model_step.coef_

    importance_df = pd.DataFrame({
        "Feature": [clean_feature_name(feature) for feature in feature_names],
        "Coefficient": coefficients,
        "Absolute Impact": np.abs(coefficients)
    })

    importance_df = importance_df.sort_values(
        by="Absolute Impact",
        ascending=False
    ).head(10)

    return importance_df.sort_values(by="Absolute Impact", ascending=True)


def render_hero():
    if os.path.exists(BANNER_PATH):
        st.image(BANNER_PATH, width="stretch")
    else:
        st.markdown(
            dedent("""
            <div class="section-card">
                <h2>Medical insurance cost prediction</h2>
                <p>
                    Predicting insurance charges using patient profile, BMI, smoking status, dependents, and region.
                </p>
            </div>
            """),
            unsafe_allow_html=True
        )


# ============================================================
# Sidebar navigation
# ============================================================

st.sidebar.markdown("### Medical insurance ML app")
st.sidebar.caption("Prediction + management insights")

page = st.sidebar.radio(
    label="Navigation",
    options=[
        "Prediction app",
        "Management insights dashboard"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Final model")
st.sidebar.write("Linear Regression")
st.sidebar.caption("Test performance")
st.sidebar.write(f"RMSE: {format_currency(metadata.get('test_rmse', 4677.1552))}")
st.sidebar.write(f"R²: {metadata.get('test_r2', 0.8532):.3f}")


# ============================================================
# Prediction page
# ============================================================

if page == "Prediction app":

    render_hero()

    st.markdown(
        dedent("""
        <div class="section-card">
            <h2>Medical insurance cost prediction using machine learning</h2>
            <p>
                This app predicts estimated medical insurance charges using the final selected model:
                <b>Linear Regression trained on original charges</b>.
                The model was selected after comparing baseline, log-transformed, and tuned regression models.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.subheader("1. Input profile")
    st.caption(
        "Enter customer profile details. The app automatically creates the same engineered features used during model training."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=64, value=35, step=1)
        sex = st.selectbox("Sex", options=["female", "male"])

    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=30.0, step=0.1)
        children = st.number_input("Number of children", min_value=0, max_value=5, value=1, step=1)

    with col3:
        smoker = st.selectbox("Smoker", options=["no", "yes"])
        region = st.selectbox("Region", options=["northeast", "northwest", "southeast", "southwest"])

    age_group = create_age_group(age)
    bmi_category = create_bmi_category(bmi)
    smoker_bmi_interaction = f"{smoker}_{bmi_category}"
    profile_summary = get_profile_summary(smoker, bmi_category, age_group)

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region,
        "age_group": age_group,
        "bmi_category": bmi_category,
        "smoker_bmi_interaction": smoker_bmi_interaction
    }])

    st.subheader("2. Derived cost-driver segment")
    st.caption("These derived fields show how raw inputs are converted into model-ready customer segments.")

    dcol1, dcol2, dcol3 = st.columns(3)

    with dcol1:
        st.markdown(
            dedent(f"""
            <div class="input-summary-card">
                <div class="input-summary-title">Age group</div>
                <div class="input-summary-value">{age_group}</div>
                <div class="input-summary-note">Derived from selected age</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with dcol2:
        st.markdown(
            dedent(f"""
            <div class="input-summary-card">
                <div class="input-summary-title">BMI category</div>
                <div class="input-summary-value">{bmi_category}</div>
                <div class="input-summary-note">Derived from selected BMI</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with dcol3:
        st.markdown(
            dedent(f"""
            <div class="input-summary-card">
                <div class="input-summary-title">Cost-driver profile</div>
                <div class="input-summary-value">{profile_summary}</div>
                <div class="input-summary-note">Uses smoker, age, and BMI pattern</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with st.expander("View model input preview"):
        st.dataframe(input_data, width="stretch")

    predict_clicked = st.button("Predict insurance charge", type="primary")

    if predict_clicked:
        prediction = float(model.predict(input_data)[0])

        rmse = metadata.get("test_rmse", 4677.1552)
        lower_range = max(0, prediction - rmse)
        upper_range = prediction + rmse

        q25 = df["charges"].quantile(0.25)
        q75 = df["charges"].quantile(0.75)
        q90 = df["charges"].quantile(0.90)

        cost_segment = get_cost_segment(prediction, q25, q75, q90)
        recommendation = get_recommendation_for_prediction(smoker, bmi_category, age_group)

        st.divider()
        st.subheader("3. Prediction")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.markdown(
                dedent(f"""
                <div class="metric-card">
                    <div class="metric-title">Predicted insurance charge</div>
                    <div class="metric-value">{format_currency(prediction)}</div>
                </div>
                """),
                unsafe_allow_html=True
            )

        with result_col2:
            st.markdown(
                dedent(f"""
                <div class="metric-card">
                    <div class="metric-title">Estimated prediction range</div>
                    <div class="metric-value metric-value-green">
                        {format_currency(lower_range)} - {format_currency(upper_range)}
                    </div>
                </div>
                """),
                unsafe_allow_html=True
            )

        with result_col3:
            st.markdown(
                dedent(f"""
                <div class="metric-card">
                    <div class="metric-title">Cost segment</div>
                    <div class="metric-value metric-value-neutral">{cost_segment}</div>
                </div>
                """),
                unsafe_allow_html=True
            )

        st.subheader("4. Management interpretation")

        st.markdown(
            dedent(f"""
            <div class="insight-card">
                <h3>Prediction insight</h3>
                <p><b>What this means:</b> The estimated charge for this profile is <b>{format_currency(prediction)}</b>.</p>
                <p><b>Prediction range:</b> {get_range_note(rmse)}</p>
                <p><b>Management takeaway:</b> {recommendation}</p>
            </div>
            """),
            unsafe_allow_html=True
        )

    st.markdown(
        dedent("""
        <div class="footer-note">
            Disclaimer: This is an educational ML prototype and should not be used for real medical,
            insurance pricing, underwriting, eligibility, or policy decisions.
        </div>
        """),
        unsafe_allow_html=True
    )


# ============================================================
# Management insights dashboard
# ============================================================

if page == "Management insights dashboard":

    render_hero()

    st.markdown(
        dedent("""
        <div class="section-card">
            <h2>Medical insurance cost management insights dashboard</h2>
            <p>
                This dashboard answers management questions about cost drivers, high-cost customer segments,
                model selection, and risk-monitoring opportunities. It is designed for insight generation,
                not real insurance pricing or underwriting decisions.
            </p>
        </div>
        """),
        unsafe_allow_html=True
    )

    avg_charge = df["charges"].mean()
    smoker_avg = df[df["smoker"] == "yes"]["charges"].mean()
    non_smoker_avg = df[df["smoker"] == "no"]["charges"].mean()
    smoker_multiplier = smoker_avg / non_smoker_avg
    final_rmse = metadata.get("test_rmse", 4677.1552)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(
            dedent(f"""
            <div class="metric-card">
                <div class="metric-title">Average charge</div>
                <div class="metric-value">{format_currency(avg_charge)}</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with kpi2:
        st.markdown(
            dedent(f"""
            <div class="metric-card">
                <div class="metric-title">Smoker avg charge</div>
                <div class="metric-value">{format_currency(smoker_avg)}</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with kpi3:
        st.markdown(
            dedent(f"""
            <div class="metric-card">
                <div class="metric-title">Smoker cost multiplier</div>
                <div class="metric-value">{smoker_multiplier:.1f}x</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    with kpi4:
        st.markdown(
            dedent(f"""
            <div class="metric-card">
                <div class="metric-title">Final model RMSE</div>
                <div class="metric-value">{format_currency(final_rmse)}</div>
            </div>
            """),
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="dashboard-question">1. Which customer segments create the highest insurance cost exposure?</div>',
        unsafe_allow_html=True
    )

    segment_summary = (
        df.groupby(["smoker", "age_group", "bmi_category"], observed=False)
        .agg(
            average_charge=("charges", "mean"),
            median_charge=("charges", "median"),
            record_count=("charges", "count")
        )
        .reset_index()
    )

    segment_summary = segment_summary[segment_summary["record_count"] >= 10].copy()
    segment_summary["segment"] = (
        "Smoker: " + segment_summary["smoker"].astype(str)
        + " | Age: " + segment_summary["age_group"].astype(str)
        + " | BMI: " + segment_summary["bmi_category"].astype(str)
    )

    top_segments = segment_summary.sort_values(
        by="average_charge",
        ascending=False
    ).head(10)

    fig_top_segments = px.bar(
        top_segments.sort_values("average_charge", ascending=True),
        x="average_charge",
        y="segment",
        orientation="h",
        text="average_charge",
        labels={
            "average_charge": "Average charge",
            "segment": "Customer segment"
        }
    )

    fig_top_segments.update_traces(
        marker_color="#8b0000",
        texttemplate="$%{text:,.0f}",
        textposition="outside"
    )

    fig_top_segments.update_layout(
        height=520,
        margin=dict(l=20, r=30, t=20, b=20),
        showlegend=False
    )

    st.plotly_chart(fig_top_segments, width="stretch")

    st.markdown(
        dedent("""
        <div class="insight-card">
            <h3>High-cost segment insight</h3>
            <p><b>What the chart says:</b> The highest average charges are concentrated in combined segments involving smoking, older age groups, and higher BMI categories.</p>
            <p><b>Management takeaway:</b> Management should monitor these segments for reserve planning, risk concentration, pricing review, and preventive wellness-program design.</p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-question">2. How much does smoking amplify cost across BMI groups?</div>',
        unsafe_allow_html=True
    )

    bmi_smoker_summary = (
        df.groupby(["bmi_category", "smoker"], observed=False)
        .agg(
            average_charge=("charges", "mean"),
            record_count=("charges", "count")
        )
        .reset_index()
    )

    fig_bmi_smoker = px.bar(
        bmi_smoker_summary,
        x="bmi_category",
        y="average_charge",
        color="smoker",
        barmode="group",
        text="average_charge",
        labels={
            "bmi_category": "BMI category",
            "average_charge": "Average charge",
            "smoker": "Smoker"
        },
        color_discrete_map={
            "yes": "#8b0000",
            "no": "#6c757d"
        }
    )

    fig_bmi_smoker.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside"
    )

    fig_bmi_smoker.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig_bmi_smoker, width="stretch")

    st.markdown(
        dedent("""
        <div class="insight-card">
            <h3>BMI and smoker interaction insight</h3>
            <p><b>What the chart says:</b> Smokers have much higher average charges across BMI categories, and the difference becomes especially visible in obese BMI groups.</p>
            <p><b>Management takeaway:</b> Smoking and BMI together form a stronger cost signal than BMI alone. This supports wellness targeting and closer monitoring of high-BMI smoker segments.</p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-question">3. Does cost exposure rise with age, and does smoking change the pattern?</div>',
        unsafe_allow_html=True
    )

    age_smoker_summary = (
        df.groupby(["age_group", "smoker"], observed=False)
        .agg(
            average_charge=("charges", "mean"),
            record_count=("charges", "count")
        )
        .reset_index()
    )

    fig_age_smoker = px.line(
        age_smoker_summary,
        x="age_group",
        y="average_charge",
        color="smoker",
        markers=True,
        labels={
            "age_group": "Age group",
            "average_charge": "Average charge",
            "smoker": "Smoker"
        },
        color_discrete_map={
            "yes": "#8b0000",
            "no": "#6c757d"
        }
    )

    fig_age_smoker.update_traces(line=dict(width=4), marker=dict(size=9))

    fig_age_smoker.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig_age_smoker, width="stretch")

    st.markdown(
        dedent("""
        <div class="insight-card">
            <h3>Age and smoker cost trend insight</h3>
            <p><b>What the chart says:</b> Average charges increase with age, but smoker profiles remain consistently much higher than non-smoker profiles.</p>
            <p><b>Management takeaway:</b> Older smoker groups create higher cost exposure and may need stronger reserve planning and proactive health engagement.</p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-question">4. Which segments may need pricing review, risk monitoring, or wellness focus?</div>',
        unsafe_allow_html=True
    )

    q75 = df["charges"].quantile(0.75)
    q90 = df["charges"].quantile(0.90)

    recommendation_matrix = top_segments.copy()
    recommendation_matrix["risk_level"] = recommendation_matrix["average_charge"].apply(
        lambda value: make_segment_risk_level(value, q75, q90)
    )

    recommendation_matrix["suggested_action"] = recommendation_matrix.apply(
        lambda row: make_segment_action(
            row["risk_level"],
            row["smoker"],
            row["bmi_category"]
        ),
        axis=1
    )

    recommendation_matrix_display = recommendation_matrix[
        [
            "segment",
            "record_count",
            "average_charge",
            "median_charge",
            "risk_level",
            "suggested_action"
        ]
    ].copy()

    recommendation_matrix_display["average_charge"] = recommendation_matrix_display["average_charge"].round(2)
    recommendation_matrix_display["median_charge"] = recommendation_matrix_display["median_charge"].round(2)

    st.dataframe(
        recommendation_matrix_display,
        width="stretch",
        hide_index=True
    )

    st.markdown(
        dedent("""
        <div class="insight-card">
            <h3>Recommendation matrix insight</h3>
            <p><b>What the table says:</b> The highest-cost segments can be translated into business actions such as risk monitoring, reserve planning, pricing review, and wellness focus.</p>
            <p><b>Management takeaway:</b> This table should be used as a decision-support view, not as an automated pricing or underwriting rule.</p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="dashboard-question">5. Which model features have the strongest impact on predicted charges?</div>',
        unsafe_allow_html=True
    )

    try:
        feature_importance = get_feature_importance_from_linear_pipeline(model)

        fig_feature_importance = px.bar(
            feature_importance,
            x="Absolute Impact",
            y="Feature",
            orientation="h",
            text="Absolute Impact",
            labels={
                "Absolute Impact": "Absolute coefficient impact",
                "Feature": "Feature"
            }
        )

        fig_feature_importance.update_traces(
            marker_color="#8b0000",
            texttemplate="%{text:,.0f}",
            textposition="outside"
        )

        fig_feature_importance.update_layout(
            height=500,
            margin=dict(l=20, r=30, t=20, b=20),
            showlegend=False
        )

        st.plotly_chart(fig_feature_importance, width="stretch")

        st.markdown(
            dedent("""
            <div class="insight-card">
                <h3>Feature impact insight</h3>
                <p><b>What the chart says:</b> The final Linear Regression model places the strongest coefficient impact on key encoded cost-driver features, especially smoking-related and high-risk segment features.</p>
                <p><b>Management takeaway:</b> Feature impact helps explain model behavior, but it should be interpreted as model influence, not medical causality.</p>
            </div>
            """),
            unsafe_allow_html=True
        )

    except Exception as error:
        st.warning("Feature impact chart could not be generated from the saved model pipeline.")
        st.write(error)

    st.markdown(
        '<div class="dashboard-question">6. Why was the final model selected for deployment?</div>',
        unsafe_allow_html=True
    )

    model_leaderboard = pd.DataFrame({
        "Model": [
            "Linear Regression - Original Charges",
            "Tuned Random Forest - Log Target",
            "Random Forest - Log Target",
            "Tuned Gradient Boosting - Log Target",
            "Gradient Boosting - Log Target"
        ],
        "Test RMSE": [
            4677.1552,
            4881.8554,
            4903.6827,
            4925.1912,
            4978.0109
        ],
        "Test R2": [
            0.8532,
            0.8400,
            0.8386,
            0.8372,
            0.8337
        ],
        "Overfitting Status": [
            "Low",
            "Low",
            "Moderate",
            "Low",
            "Low"
        ]
    })

    fig_model_leaderboard = px.bar(
        model_leaderboard.sort_values("Test RMSE", ascending=True),
        x="Model",
        y="Test RMSE",
        text="Test RMSE",
        labels={
            "Model": "Model",
            "Test RMSE": "Test RMSE"
        }
    )

    fig_model_leaderboard.update_traces(
        marker_color="#8b0000",
        texttemplate="$%{text:,.0f}",
        textposition="outside"
    )

    fig_model_leaderboard.update_layout(
        height=460,
        margin=dict(l=20, r=20, t=20, b=80),
        xaxis_tickangle=-20
    )

    st.plotly_chart(fig_model_leaderboard, width="stretch")
    st.dataframe(model_leaderboard, width="stretch", hide_index=True)

    st.markdown(
        dedent("""
        <div class="insight-card">
            <h3>Model selection insight</h3>
            <p><b>What the chart says:</b> Linear Regression trained on original charges has the lowest Test RMSE and highest Test R² among the top models.</p>
            <p><b>Management takeaway:</b> The selected model balances accuracy, low overfitting, simplicity, and deployment stability. The model should be used as an educational decision-support prototype, not an automated pricing engine.</p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown(
        dedent("""
        <div class="footer-note">
            Disclaimer: This dashboard is an educational ML prototype. It should not be used for real medical,
            insurance pricing, underwriting, eligibility, or policy decisions. Recommendations are for analytical
            interpretation only.
        </div>
        """),
        unsafe_allow_html=True
    )
