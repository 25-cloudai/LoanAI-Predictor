import streamlit as st
import pandas as pd
import joblib
from streamlit_option_menu import option_menu

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="",
    page_icon="🏦",
    layout="wide"
)
st.markdown("""
<style>

.stApp{
background:linear-gradient(
135deg,
#020617,
#0f172a,
#1e293b
);
color:white;
}

.hero-card{
padding:80px;
border-radius:30px;
text-align:center;
background:linear-gradient(
135deg,
#2563eb,
#1e40af,
#1e3a8a
);
color:white;
box-shadow:0 0 30px rgba(37,99,235,0.4);
}

.feature-card{
background:#0f172a;
padding:30px;
border-radius:20px;
border:1px solid #334155;
text-align:center;
height:250px;
}

.feature-card:hover{
transform:translateY(-8px);
transition:0.4s;
}

.logo{
font-size:70px;
}

</style>
""", unsafe_allow_html=True)
# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("models/loan_model.pkl")
encoders = joblib.load("models/encoders.pkl")

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    padding-top: 0rem;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 12px;
    font-size: 18px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# NAVBAR
# =====================================

selected = option_menu(
    menu_title=None,
    options=[
    "Home",
    "Loan Prediction",
    "Analytics",
    "About"
    ],
    icons=[
    "house",
    "calculator",
    "bar-chart",
    "info-circle"
    ],
    orientation="horizontal"
)

# =====================================
# HOME PAGE
# =====================================

if selected == "Home":

    st.markdown("""
    <div class="hero-card">

    <div class="logo">🏦</div>

    <h1 style="font-size:80px;">
    LoanAI
    </h1>

    <h3>
    AI-Powered Loan Risk Assessment Platform
    </h3>

    <p style="font-size:22px;">
    Smart Loan Decisions Using Artificial Intelligence
    </p>

    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1,c2,c3,c4 = st.columns(4)

    c1.metric("Customers", "20,000+")
    c2.metric("Features", "17")
    c3.metric("Accuracy", "85%")
    c4.metric("Predictions", "50K+")

    st.markdown("---")

    st.markdown("## Why Choose LoanAI?")

    col1,col2,col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
        <h1>🤖</h1>
        <h3>AI Prediction</h3>
        <p>
        Advanced machine learning predicts
        repayment probability instantly.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
        <h1>📊</h1>
        <h3>Risk Analysis</h3>
        <p>
        Identify high-risk customers
        before loan approval.
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
        <h1>⚡</h1>
        <h3>Instant Results</h3>
        <p>
        Get accurate predictions
        within seconds.
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## How LoanAI Works")

    s1,s2,s3,s4 = st.columns(4)

    s1.success("1️⃣ Enter Data")
    s2.info("2️⃣ AI Processing")
    s3.warning("3️⃣ Risk Evaluation")
    s4.error("4️⃣ Prediction")

    st.markdown("---")

    if st.button("🚀 Go To Loan Prediction", use_container_width=True):
        st.info("Click Loan Prediction in the Navbar Above ⬆️")

# =====================================
# LOAN PREDICTION PAGE
# =====================================

elif selected == "Loan Prediction":

    st.markdown("""
    # 🔍 Loan Prediction

    Evaluate customer loan repayment probability
    using our trained AI model.
    """)

    # Hide form initially
    if "show_form" not in st.session_state:
        st.session_state.show_form = False

    if not st.session_state.show_form:
        if st.button(
            "🚀 Start Prediction",
            use_container_width=True
        ):
            st.session_state.show_form = True
            st.rerun()

    if st.session_state.show_form:

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            age = st.selectbox(
                "Age",
                ["Select Age"] + list(range(18, 101))
            )

            gender = st.selectbox(
                "Gender",
                [
                    "Select Gender",
                    "Male",
                    "Female",
                    "Other"
                ]
            )

            marital_status = st.selectbox(
                "Marital Status",
                [
                    "Select Marital Status",
                    "Married",
                    "Single",
                    "Divorced",
                    "Widowed"
                ]
            )

            education_level = st.selectbox(
                "Education Level",
                [
                    "Select Education Level",
                    "Master's",
                    "Bachelor's",
                    "High School",
                    "Other",
                    "PhD"
                ]
            )

            annual_income = st.slider(
                "Annual Income (₹)",
                50000,
                2000000,
                500000,
                50000
            )

            monthly_income = st.selectbox(
                "Monthly Income (₹)",
                list(range(5000, 200001, 5000))
            )

            employment_status = st.selectbox(
                "Employment Status",
                [
                    "Select Employment Status",
                    "Employed",
                    "Unemployed",
                    "Self-employed",
                    "Student",
                    "Retired"
                ]
            )

            debt_to_income_ratio = st.slider(
                "Debt To Income Ratio (%)",
                0.0,
                100.0,
                25.0
            )

        with col2:

            credit_score = st.slider(
                "Credit Score",
                300,
                900,
                700
            )

            loan_amount = st.selectbox(
                "Loan Amount (₹)",
                list(range(10000, 5000001, 10000))
            )

            loan_purpose = st.selectbox(
                "Loan Purpose",
                [
                    "Select Loan Purpose",
                    "Car",
                    "Debt consolidation",
                    "Business",
                    "Other",
                    "Home",
                    "Medical",
                    "Education",
                    "Vacation"
                ]
            )

            interest_rate = st.slider(
                "Interest Rate (%)",
                1.0,
                30.0,
                10.0
            )

            loan_term = st.selectbox(
                "Loan Term (Months)",
                [12, 24, 36, 48, 60, 84, 120]
            )

            installment = st.selectbox(
                "Monthly Installment (₹)",
                list(range(1000, 100001, 1000))
            )

            num_of_open_accounts = st.number_input(
                "Open Accounts",
                min_value=0
            )

            total_credit_limit = st.selectbox(
                "Total Credit Limit (₹)",
                list(range(50000, 5000001, 50000))
            )

            current_balance = st.selectbox(
                "Current Balance (₹)",
                list(range(0, 2000001, 10000))
            )

        if st.button(
            "🔮 Analyze Loan Risk",
            use_container_width=True
        ):

            if (
                age == "Select Age"
                or gender == "Select Gender"
                or marital_status == "Select Marital Status"
                or education_level == "Select Education Level"
                or employment_status == "Select Employment Status"
                or loan_purpose == "Select Loan Purpose"
            ):
                st.error("⚠️ Please fill all required fields.")
                st.stop()

            input_data = pd.DataFrame([[

                age,
                encoders["gender"].transform([gender])[0],
                encoders["marital_status"].transform([marital_status])[0],
                encoders["education_level"].transform([education_level])[0],
                annual_income,
                monthly_income,
                encoders["employment_status"].transform([employment_status])[0],
                debt_to_income_ratio,
                credit_score,
                loan_amount,
                encoders["loan_purpose"].transform([loan_purpose])[0],
                interest_rate,
                loan_term,
                installment,
                num_of_open_accounts,
                total_credit_limit,
                current_balance

            ]], columns=[

                'age',
                'gender',
                'marital_status',
                'education_level',
                'annual_income',
                'monthly_income',
                'employment_status',
                'debt_to_income_ratio',
                'credit_score',
                'loan_amount',
                'loan_purpose',
                'interest_rate',
                'loan_term',
                'installment',
                'num_of_open_accounts',
                'total_credit_limit',
                'current_balance'
            ])

            prediction = model.predict(input_data)[0]
            probs = model.predict_proba(input_data)[0]

            repay_prob = probs[1] * 100
            default_prob = probs[0] * 100

            st.markdown("---")

            if prediction == 1:
                st.success("✅ Customer is Likely to Repay")
            else:
                st.error("❌ Customer is High Risk")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Repayment Probability",
                    f"{repay_prob:.2f}%"
                )

            with col2:
                st.metric(
                    "Default Probability",
                    f"{default_prob:.2f}%"
                )

            st.progress(repay_prob / 100)

            if repay_prob >= 80:
                st.success("🟢 Very Low Risk")

            elif repay_prob >= 60:
                st.info("🟡 Low Risk")

            elif repay_prob >= 40:
                st.warning("🟠 Medium Risk")

            else:
                st.error("🔴 High Risk")
         
elif selected == "Analytics":

    st.title("📊 Analytics Dashboard")

    importance_df = pd.read_csv(
        "models/feature_importance.csv"
    )

    st.subheader("Feature Importance")

    st.bar_chart(
        importance_df.set_index("Feature")
    )

    st.dataframe(
        importance_df,
        use_container_width=True
    )

# =====================================
# ABOUT PAGE
# =====================================

elif selected == "About":

    st.markdown("""
    # ℹ️ About LoanAI

    ### Project Overview

    LoanAI is an AI-powered loan repayment
    prediction platform built using Machine Learning.

    ### Technologies Used

    - Python
    - Pandas
    - NumPy
    - Scikit-Learn
    - SMOTE
    - Random Forest
    - Streamlit

    ### Dataset

    - 20,000 Records
    - 17 Features

    ### Model Performance

    - Accuracy: 85%
    - Balanced using SMOTE
    - Random Forest Classifier

    ### Objective

    Assist financial institutions in evaluating
    loan repayment risk and making better
    credit decisions.
    """)