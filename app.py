import streamlit as st
import pickle
import numpy as np
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="centered"
)

# -------------------------------
# Load Model & Scaler (Cloud-safe)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "credit_default_rf_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# -------------------------------
# App Header
# -------------------------------
st.title("💳 Credit Default Prediction System")
st.markdown(
    """
    This application predicts the **probability of loan default**
    using a machine learning model trained on historical credit data.
    """
)

st.divider()

# -------------------------------
# Input Section
# -------------------------------
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    limit_bal = st.number_input(
        "Credit Limit (₹)",
        min_value=0.0,
        step=1000.0,
        help="Total approved credit limit for the applicant"
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        help="Applicant age in years"
    )

pay_0 = st.selectbox(
    "Repayment Status (PAY_0)",
    options=list(range(-2, 9)),
    help="""
    -2 = No consumption  
    -1 = Paid duly  
     0 = Revolving credit  
     1–8 = Payment delay (months)
    """
)

# -------------------------------
# Prediction
# -------------------------------
st.divider()

if st.button("🔍 Predict Credit Risk"):
    X = np.array([[limit_bal, age, pay_0]])
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        label="Default Probability",
        value=f"{probability:.2%}"
    )

    if probability >= 0.6:
        st.error("⚠️ High Credit Risk: Likely to Default")
        st.markdown(
            """
            **Recommendation:**  
            - Consider rejecting the loan  
            - Or apply stricter terms / higher interest
            """
        )
    else:
        st.success("✅ Low Credit Risk: Unlikely to Default")
        st.markdown(
            """
            **Recommendation:**  
            - Applicant appears creditworthy  
            - Loan approval can be considered
            """
        )

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.caption("⚙️ ML Model: Random Forest | Key Metrics: Recall & F1-score")
