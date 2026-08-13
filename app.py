import streamlit as st
import pandas as pd
import torch
import joblib
from model_def import ChurnNetV3

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("models/preprocessor.pkl")
    input_dim = joblib.load("models/input_dim.pkl")
    model = ChurnNetV3(input_dim)
    model.load_state_dict(torch.load("models/churn_model.pth", map_location="cpu"))
    model.eval()
    return preprocessor, model

preprocessor, model = load_artifacts()

st.title("📉 Customer Churn Predictor")
st.write("مشخصات مشتری را وارد کنید تا احتمال ریزش (Churn) پیش‌بینی شود.")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partner", ["Yes", "No"])
    Dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

with col2:
    DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

if st.button("پیش‌بینی کن"):
    input_df = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
        "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
        "MultipleLines": MultipleLines, "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
        "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
        "Contract": Contract, "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    X_input = preprocessor.transform(input_df)
    X_input = X_input.toarray() if hasattr(X_input, "toarray") else X_input
    X_tensor = torch.tensor(X_input, dtype=torch.float32)

    with torch.no_grad():
        logit = model(X_tensor)
        prob = torch.sigmoid(logit).item()

    st.subheader(f"احتمال ریزش: {prob*100:.1f}%")
    if prob >= 0.5:
        st.error("⚠️ این مشتری در معرض ریزش (Churn) است.")
    else:
        st.success("✅ این مشتری احتمالاً می‌ماند.")
