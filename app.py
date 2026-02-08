import streamlit as st
import pickle
import numpy as np
import os

# Load model and scaler using absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(
    open(os.path.join(BASE_DIR, "model", "credit_default_rf_model.pkl"), "rb")
)
scaler = pickle.load(
    open(os.path.join(BASE_DIR, "model", "scaler.pkl"), "rb")
)

st.set_page_config(page_title="Credit Default Prediction")

st.title("💳 Credit Default Prediction")
st.write("Enter customer details to predict default risk")

limit_bal = st.number_input("Credit Limit", min_value=0.0, step=1000.0)
age = st.number_input("Age", min_value=18, max_value=100)
pay_0 = st.number_input("Repayment Status (PAY_0)", min_value=-2, max_value=8)

if st.button("Predict"):
    X = np.array([[limit_bal, age, pay_0]])
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)

    if prediction[0] == 1:
        st.error("⚠️ High Risk: Likely to Default")
    else:
        st.success("✅ Low Risk: Not Likely to Default")
