
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111c31;
    border-right: 1px solid #26364f;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
p, label, span {
    color: #cbd5e1;
}
.metric-card {
    background: linear-gradient(135deg, #172943, #142238);
    padding: 22px;
    border: 1px solid #29415f;
    border-radius: 14px;
    margin-bottom: 12px;
}
.result-card {
    background: linear-gradient(135deg, #172943, #1c3150);
    border: 1px solid #3b82f6;
    border-radius: 16px;
    padding: 25px;
    margin-top: 20px;
}
.small-text {
    color: #94a3b8;
    font-size: 13px;
}
div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    border-radius: 9px;
    height: 48px;
    font-weight: bold;
    width: 100%;
}
div.stButton > button:hover {
    border: 1px solid #93c5fd;
    color: white;
}
hr {
    border-color: #26364f;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("# ChurnIQ")
    st.caption("CUSTOMER ANALYTICS PLATFORM")
    st.divider()

    st.markdown("### Customer Profile")

    with st.form("customer_form"):

        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

        st.markdown("### Subscription Details")

        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        st.markdown("### Billing Information")

        monthly = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

        predict_button = st.form_submit_button(
            "Analyze Customer"
        )

# ---------------- MAIN DASHBOARD ----------------
st.markdown(
    "<p class='small-text'>MACHINE LEARNING / CUSTOMER RETENTION / ANALYTICS</p>",
    unsafe_allow_html=True
)

st.title("Customer Churn Intelligence")
st.write(
    "Predict customer retention risk using machine learning "
    "and identify opportunities to improve customer loyalty."
)

st.divider()

# KPI CARDS
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <p class="small-text">MODEL TYPE</p>
        <h3>Logistic Regression</h3>
        <p class="small-text">Balanced Class Weights</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <p class="small-text">MODEL ACCURACY</p>
        <h3>74%</h3>
        <p class="small-text">Test dataset performance</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <p class="small-text">CHURN RECALL</p>
        <h3>78%</h3>
        <p class="small-text">Actual churn customers identified</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## Customer Risk Analysis")

if predict_button:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    customer_encoded = pd.get_dummies(
        customer,
        drop_first=False,
        dtype=int
    )

    customer_encoded = customer_encoded.reindex(
        columns=model_columns,
        fill_value=0
    )

    numerical_cols = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    customer_encoded[numerical_cols] = scaler.transform(
        customer_encoded[numerical_cols]
    )

    prediction = model.predict(customer_encoded)[0]
    probability = model.predict_proba(customer_encoded)[0][1]

    # st.markdown('<div class="result-card">', unsafe_allow_html=True)

    if prediction == 1:
        st.error("HIGH RISK: CUSTOMER LIKELY TO CHURN")
        st.markdown("### Retention action recommended")
        st.write(
            "This customer is classified as likely to leave. "
            "Consider proactive engagement and a suitable retention offer."
        )
    else:
        st.success("LOW RISK: CUSTOMER LIKELY TO STAY")
        st.markdown("### Customer retention outlook")
        st.write(
            "This customer is classified as likely to stay. "
            "Continue delivering a positive service experience."
        )

    st.markdown("### Churn Probability")

    st.progress(float(probability))
    st.write(f"**{probability:.1%}** estimated churn probability")

    # st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Customer Summary")

    a, b, c = st.columns(3)

    a.metric("Tenure", f"{tenure} months")
    b.metric("Monthly Charges", f"${monthly:.2f}")
    c.metric("Contract", contract)

else:
    st.info(
        "Enter customer details in the sidebar and click "
        "'Analyze Customer' to generate a prediction."
    )

    st.markdown("""
    <div class="metric-card">
        <h3>How it works</h3>
        <p>01 — Enter customer subscription and billing details.</p>
        <p>02 — Machine learning model analyzes the customer profile.</p>
        <p>03 — View churn probability and retention insights.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.markdown("### About This Project")
st.write(
    "An end-to-end customer churn prediction project built using "
    "Python, Pandas, Scikit-learn and Streamlit. "
    "The model uses customer demographics, subscription details "
    "and billing information to estimate churn risk."
)

st.caption("ChurnIQ | Machine Learning Project | Customer Retention Analytics")
