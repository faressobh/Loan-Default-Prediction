import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# Page Configuration & Styling
# ============================================================
st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="💳",
    layout="wide"
)

# Load the model, scaler, and feature names
@st.cache_resource
def load_artifacts():
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_artifacts()

# ============================================================
# Header Section with Banking Image & Intro
# ============================================================
col_header1, col_header2 = st.columns([2, 1])

with col_header1:
    st.title("💳 Loan Default Risk Prediction System")
    st.markdown("""
        Welcome to the **Loan Risk Evaluation Portal**. 
        Please fill in your financial and personal details below to instantly check your loan approval risk assessment. 
        Our intelligent system helps secure safe and reliable banking decisions.
    """)

with col_header2:

    st.image("bank.jpg", caption="Secure Banking & Finance", use_container_width=True)

st.markdown("---")

# ============================================================
# User Input Form (Organized in Columns with Descriptions Above Inputs)
# ============================================================
st.subheader("📋 Enter Your Financial Details")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Revolving Utilization
        st.markdown("##### 🔹 Revolving Utilization Of Unsecured Lines")
        st.caption("The total balance on your credit cards and lines of credit divided by your total credit limit (kept between 0 and 1).")
        revolving_util = st.number_input(
            "Revolving Utilization Value", min_value=0.0, max_value=1.0, value=0.3, label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 3. Past Due 30-59 Days
        st.markdown("##### 🔹 Number Of Time 30-59 Days Past Due Not Worse")
        st.caption("How many times you have been late on a payment by 30 to 59 days in the past 2 years.")
        past_due_30_59 = st.number_input(
            "30-59 Days Past Due Count", min_value=0, value=0, label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 5. Monthly Income
        st.markdown("##### 🔹 Monthly Income")
        st.caption("Your total monthly income in dollars.")
        monthly_income = st.number_input(
            "Monthly Income Amount ($)", min_value=0.0, value=5000.0, label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 7. Times 90 Late
        st.markdown("##### 🔹 Number Of Times 90 Days Late")
        st.caption("How many times you have been late on a payment by 90 days or more.")
        times_90_late = st.number_input(
            "90+ Days Late Count", min_value=0, value=0, label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 9. Past Due 60-89 Days
        st.markdown("##### 🔹 Number Of Time 60-89 Days Past Due Not Worse")
        st.caption("How many times you have been late on a payment by 60 to 89 days in the past 2 years.")
        past_due_60_89 = st.number_input(
            "60-89 Days Past Due Count", min_value=0, value=0, label_visibility="collapsed"
        )

    with col2:
        # 2. Age
        st.markdown("##### 🔹 Age")
        st.caption("Your current age in years.")
        age = st.number_input(
            "Age in Years", min_value=18, max_value=100, value=35, label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 4. Debt Ratio
        st.markdown("##### 🔹 Debt Ratio")
        st.caption("Your monthly debt payments divided by your gross monthly income.")
        debt_ratio = st.number_input(
            "Debt Ratio Value", min_value=0.0, value=0.3, label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 6. Open Credit Lines
        st.markdown("##### 🔹 Number Of Open Credit Lines And Loans")
        st.caption("The total number of open loans (like car loans or mortgages) and active credit cards.")
        open_credit_lines = st.number_input(
            "Open Credit Lines Count", min_value=0, value=5, label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 8. Real Estate Loans
        st.markdown("##### 🔹 Number Real Estate Loans Or Lines")
        st.caption("Number of mortgage loans and real estate credit lines you currently have.")
        real_estate_loans = st.number_input(
            "Real Estate Loans Count", min_value=0, value=1, label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # 10. Dependents
        st.markdown("##### 🔹 Number Of Dependents")
        st.caption("The number of family members or dependents living with you who rely on your financial support.")
        dependents = st.number_input(
            "Dependents Count", min_value=0, value=0, label_visibility="collapsed"
        )

    st.markdown("")
    submit_button = st.form_submit_button(label="🔍 Predict Default Risk", use_container_width=True)

# ============================================================
# Prediction Logic
# ============================================================
if submit_button:
    # Apply the same log(x + 1) transform used on MonthlyIncome during training,
    # so the value the model sees is on the same scale it was trained on.
    monthly_income_log = np.log1p(monthly_income + 1)

    # Build the input row in the same column order the model was trained on
    input_data = pd.DataFrame([{
        'RevolvingUtilizationOfUnsecuredLines': revolving_util,
        'age': age,
        'NumberOfTime30-59DaysPastDueNotWorse': past_due_30_59,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income_log,
        'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
        'NumberOfTimes90DaysLate': times_90_late,
        'NumberRealEstateLoansOrLines': real_estate_loans,
        'NumberOfTime60-89DaysPastDueNotWorse': past_due_60_89,
        'NumberOfDependents': dependents
    }])

    # Make sure the column order exactly matches the training order
    input_data = input_data[feature_names]

    # Process and Predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Default** (Probability: {probability:.1%})\n")
    else:
        st.success(f"✅ **Low Risk of Default** (Probability of Default: {probability:.1%})\n")
