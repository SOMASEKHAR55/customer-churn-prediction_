
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="ChurnIQ | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- FILE PATHS ----------------
BASE_DIR = Path(__file__).resolve().parent

EXPECTED_CSV = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


# ---------------- FIND DATASET AUTOMATICALLY ----------------
def find_dataset():
    csv_files = list(BASE_DIR.rglob("*.csv"))

    # 1. Exact filename match
    for file in csv_files:
        if file.name.lower() == EXPECTED_CSV.lower():
            return file

    # 2. Match filename containing Telco and Churn
    for file in csv_files:
        if "telco" in file.name.lower() and "churn" in file.name.lower():
            return file

    # 3. Use the only CSV if there is just one
    if len(csv_files) == 1:
        return csv_files[0]

    return None


DATA_PATH = find_dataset()


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)

    required_columns = [
        "Churn",
        "Contract",
        "PaymentMethod",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {missing_columns}"
        )

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    df.dropna(subset=["TotalCharges"], inplace=True)

    return df


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_files():
    model = joblib.load(BASE_DIR / "churn_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    model_columns = joblib.load(BASE_DIR / "model_columns.pkl")

    return model, scaler, model_columns


model, scaler, model_columns = load_model_files()


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
    padding: 20px;
    border: 1px solid #29415f;
    border-radius: 14px;
    margin-bottom: 12px;
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

hr {
    border-color: #26364f;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR NAVIGATION ----------------
with st.sidebar:
    st.markdown("# ChurnIQ")
    st.caption("CUSTOMER ANALYTICS PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Analytics Dashboard", "Churn Prediction"]
    )

    st.divider()
    st.caption("Powered by Python | Machine Learning | Streamlit")


# =====================================================
#                ANALYTICS DASHBOARD
# =====================================================

if page == "Analytics Dashboard":

    st.markdown(
        "<p class='small-text'>CUSTOMER RETENTION / BUSINESS INTELLIGENCE</p>",
        unsafe_allow_html=True
    )

    st.title("Customer Analytics Dashboard")

    st.write(
        "Explore customer behavior, identify churn patterns "
        "and discover opportunities to improve customer retention."
    )

    st.divider()

    # CHECK DATASET
    if DATA_PATH is None:

        st.error("Customer churn CSV file not found.")

        csv_files = list(BASE_DIR.rglob("*.csv"))

        st.write("CSV files detected in repository:")

        st.write([
            str(file.relative_to(BASE_DIR))
            for file in csv_files
        ])

        st.info(
            "Upload the Telco Customer Churn CSV file "
            "to your GitHub repository."
        )

        st.stop()

    # LOAD DATASET
    try:
        df = load_data(str(DATA_PATH))

    except Exception as e:
        st.error(f"Dataset loading error: {e}")
        st.stop()

    # SHOW DATASET LOCATION
    with st.expander("Dataset Information"):
        st.write("Dataset location:", str(DATA_PATH.relative_to(BASE_DIR)))
        st.write("Dataset shape:", df.shape)

    # ---------------- KPI CALCULATIONS ----------------
    total_customers = len(df)

    churned_customers = (
        df["Churn"] == "Yes"
    ).sum()

    active_customers = (
        df["Churn"] == "No"
    ).sum()

    churn_rate = (
        churned_customers / total_customers * 100
        if total_customers else 0
    )

    # ---------------- KPI CARDS ----------------
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        f"{total_customers:,}"
    )

    c2.metric(
        "Churned Customers",
        f"{churned_customers:,}"
    )

    c3.metric(
        "Active Customers",
        f"{active_customers:,}"
    )

    c4.metric(
        "Overall Churn Rate",
        f"{churn_rate:.2f}%"
    )

    st.divider()

    # ---------------- CHURN DISTRIBUTION ----------------
    st.subheader("Customer Churn Distribution")

    col1, col2 = st.columns(2)

    with col1:

        churn_counts = (
            df["Churn"]
            .value_counts()
            .rename_axis("Customer Status")
            .reset_index(name="Customers")
        )

        fig = px.pie(
            churn_counts,
            names="Customer Status",
            values="Customers",
            hole=0.55,
            title="Churn vs Retained Customers"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ---------------- CONTRACT ANALYSIS ----------------
    with col2:

        contract_data = (
            df.groupby("Contract", as_index=False)
            .agg(
                Total_Customers=("Churn", "count"),
                Churned_Customers=(
                    "Churn",
                    lambda x: (x == "Yes").sum()
                )
            )
        )

        contract_data["Churn_Rate"] = (
            contract_data["Churned_Customers"]
            / contract_data["Total_Customers"]
            * 100
        )

        fig = px.bar(
            contract_data,
            x="Contract",
            y="Churn_Rate",
            color="Contract",
            text="Churn_Rate",
            title="Churn Rate by Contract Type",
            labels={
                "Churn_Rate": "Churn Rate (%)"
            }
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%"
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120",
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.divider()

    # ---------------- PAYMENT METHOD ANALYSIS ----------------
    st.subheader("Payment Method Analysis")

    payment_data = (
        df.groupby("PaymentMethod", as_index=False)
        .agg(
            Total_Customers=("Churn", "count"),
            Churned_Customers=(
                "Churn",
                lambda x: (x == "Yes").sum()
            )
        )
    )

    payment_data["Churn_Rate"] = (
        payment_data["Churned_Customers"]
        / payment_data["Total_Customers"]
        * 100
    )

    fig = px.bar(
        payment_data,
        x="PaymentMethod",
        y="Churn_Rate",
        color="Churn_Rate",
        text="Churn_Rate",
        title="Customer Churn by Payment Method",
        labels={
            "Churn_Rate": "Churn Rate (%)"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        xaxis_tickangle=-20,
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ---------------- TENURE ANALYSIS ----------------
    st.subheader("Customer Tenure Analysis")

    tenure_data = df.copy()

    tenure_data["Tenure Group"] = pd.cut(
        tenure_data["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 Months",
            "13-24 Months",
            "25-48 Months",
            "49-72 Months"
        ]
    )

    tenure_summary = (
        tenure_data.groupby(
            "Tenure Group",
            observed=False
        )
        .agg(
            Total_Customers=("Churn", "count"),
            Churned_Customers=(
                "Churn",
                lambda x: (x == "Yes").sum()
            )
        )
        .reset_index()
    )

    tenure_summary["Churn_Rate"] = (
        tenure_summary["Churned_Customers"]
        / tenure_summary["Total_Customers"].replace(
            0, float("nan")
        )
        * 100
    )

    fig = px.bar(
        tenure_summary,
        x="Tenure Group",
        y="Churn_Rate",
        color="Churn_Rate",
        text="Churn_Rate",
        title="Churn Rate Across Customer Tenure",
        labels={
            "Churn_Rate": "Churn Rate (%)"
        }
    )

    fig.update_traces(
        texttemplate="%{text:.1f}%"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120",
        coloraxis_showscale=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ---------------- MONTHLY CHARGES ANALYSIS ----------------
    st.subheader("Monthly Charges vs Customer Churn")

    fig = px.box(
        df,
        x="Churn",
        y="MonthlyCharges",
        color="Churn",
        title="Monthly Charges Distribution by Churn Status",
        labels={
            "Churn": "Customer Status",
            "MonthlyCharges": "Monthly Charges ($)"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#0b1120"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # ---------------- BUSINESS INSIGHTS ----------------
    st.subheader("Business Insights & Recommendations")

    highest_contract = contract_data.loc[
        contract_data["Churn_Rate"].idxmax()
    ]

    highest_payment = payment_data.loc[
        payment_data["Churn_Rate"].idxmax()
    ]

    st.markdown(f"""
    <div class="metric-card">
        <h3>Insight 1: Contract Risk</h3>
        <p>
        {highest_contract['Contract']} customers have the highest
        churn rate of {highest_contract['Churn_Rate']:.2f}%.
        </p>
        <p><b>Recommendation:</b> Introduce long-term contract
        incentives and customer loyalty benefits.
        </p>
    </div>

    <div class="metric-card">
        <h3>Insight 2: Payment Behavior</h3>
        <p>
        {highest_payment['PaymentMethod']} customers have the highest
        churn rate of {highest_payment['Churn_Rate']:.2f}%.
        </p>
        <p><b>Recommendation:</b> Review payment experience and
        offer convenient automatic payment options.
        </p>
    </div>

    <div class="metric-card">
        <h3>Insight 3: Customer Retention</h3>
        <p>
        Overall customer churn rate is {churn_rate:.2f}%.
        </p>
        <p><b>Recommendation:</b> Monitor high-risk customers and
        develop targeted customer retention campaigns.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---------------- DATA PREVIEW ----------------
    with st.expander("View Customer Dataset"):

        st.dataframe(
            df,
            use_container_width=True
        )

    st.caption(
        "ChurnIQ | Customer Analytics | Python & Plotly"
    )


# =====================================================
#                  CHURN PREDICTION
# =====================================================

elif page == "Churn Prediction":

    st.markdown(
        "<p class='small-text'>MACHINE LEARNING / CUSTOMER RETENTION</p>",
        unsafe_allow_html=True
    )

    st.title("Customer Churn Intelligence")

    st.write(
        "Predict customer retention risk using machine learning "
        "and identify opportunities to improve customer loyalty."
    )

    st.divider()

    # ---------------- CUSTOMER INPUT FORM ----------------
    with st.sidebar:

        st.markdown("### Customer Profile")

        with st.form("customer_form"):

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Senior Citizen",
                [0, 1]
            )

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"]
            )

            st.markdown("### Subscription Details")

            tenure = st.slider(
                "Tenure (Months)",
                0, 72, 12
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

            phone = st.selectbox(
                "Phone Service",
                ["Yes", "No"]
            )

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

            paperless = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

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

    # ---------------- MODEL INFORMATION ----------------
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Model Type",
        "Logistic Regression"
    )

    col2.metric(
        "Model Accuracy",
        "74%"
    )

    col3.metric(
        "Churn Recall",
        "78%"
    )

    st.divider()

    st.subheader("Customer Risk Analysis")

    # ---------------- PREDICTION ----------------
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

        prediction = model.predict(
            customer_encoded
        )[0]

        probability = model.predict_proba(
            customer_encoded
        )[0][1]

        if prediction == 1:

            st.error(
                "HIGH RISK: CUSTOMER LIKELY TO CHURN"
            )

            st.markdown(
                "### Retention Action Recommended"
            )

            st.write(
                "This customer is classified as likely to leave. "
                "Consider proactive engagement and a suitable retention offer."
            )

        else:

            st.success(
                "LOW RISK: CUSTOMER LIKELY TO STAY"
            )

            st.markdown(
                "### Customer Retention Outlook"
            )

            st.write(
                "This customer is classified as likely to stay. "
                "Continue delivering a positive service experience."
            )

        st.markdown("### Churn Probability")

        st.progress(
            float(probability)
        )

        st.write(
            f"**{probability:.1%}** estimated churn probability"
        )

        st.markdown("### Customer Summary")

        a, b, c = st.columns(3)

        a.metric(
            "Tenure",
            f"{tenure} months"
        )

        b.metric(
            "Monthly Charges",
            f"${monthly:.2f}"
        )

        c.metric(
            "Contract",
            contract
        )

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

    # ---------------- ABOUT PROJECT ----------------
    st.markdown("### About This Project")

    st.write(
        "An end-to-end customer churn prediction project built using "
        "Python, Pandas, Scikit-learn, Plotly and Streamlit. "
        "The model uses customer demographics, subscription details "
        "and billing information to estimate churn risk."
    )

    st.caption(
        "ChurnIQ | Machine Learning Project | Customer Retention Analytics"
    )
