
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="wide")

@st.cache_resource
def load_model():
    model  = joblib.load("model/heart_disease_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    cols   = joblib.load("model/feature_cols.pkl")
    return model, scaler, cols

model, scaler, feature_cols = load_model()

st.title("❤️ Heart Disease Prediction")
st.markdown("> **Model:** Tuned XGBoost | **Explainability:** SHAP | **Dataset:** UCI Cleveland")
st.markdown("---")

st.sidebar.header("🩺 Patient Information")
age      = st.sidebar.slider("Age", 20, 80, 55)
sex      = st.sidebar.selectbox("Sex", [("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
cp       = st.sidebar.selectbox("Chest Pain Type", [(0,"Typical Angina"),(1,"Atypical Angina"),(2,"Non-Anginal"),(3,"Asymptomatic")], format_func=lambda x: x[1])[0]
trestbps = st.sidebar.slider("Resting Blood Pressure (mmHg)", 80, 200, 130)
chol     = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 240)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120", [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])[0]
restecg  = st.sidebar.selectbox("Resting ECG", [(0,"Normal"),(1,"ST-T Abnormality"),(2,"LV Hypertrophy")], format_func=lambda x: x[1])[0]
thalach  = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
exang    = st.sidebar.selectbox("Exercise Induced Angina", [(0,"No"),(1,"Yes")], format_func=lambda x: x[1])[0]
oldpeak  = st.sidebar.slider("ST Depression", 0.0, 6.5, 1.0, step=0.1)
slope    = st.sidebar.selectbox("Slope of ST Segment", [(0,"Upsloping"),(1,"Flat"),(2,"Downsloping")], format_func=lambda x: x[1])[0]
ca       = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0,1,2,3])
thal     = st.sidebar.selectbox("Thalassemia", [(1,"Normal"),(2,"Fixed Defect"),(3,"Reversible Defect")], format_func=lambda x: x[1])[0]

age_group     = int(pd.cut([age], bins=[0,40,55,70,100], labels=[0,1,2,3])[0])
chol_bp_ratio = chol / (trestbps + 1e-5)
hr_age_ratio  = thalach / (age + 1e-5)
high_risk     = int(age > 55 and sex == 1 and cp == 3)

input_dict = {
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
    "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
    "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal,
    "age_group": age_group, "chol_bp_ratio": chol_bp_ratio,
    "hr_age_ratio": hr_age_ratio, "high_risk": high_risk
}

input_df = pd.DataFrame([input_dict])[feature_cols]

col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Patient Summary")
    display = pd.DataFrame({
        "Feature": ["Age","Sex","Chest Pain","Blood Pressure","Cholesterol","Max Heart Rate","ST Depression","Major Vessels","Thalassemia"],
        "Value": [age, "Male" if sex==1 else "Female",
                  ["Typical","Atypical","Non-Anginal","Asymptomatic"][cp],
                  f"{trestbps} mmHg", f"{chol} mg/dl", thalach, oldpeak, ca,
                  ["","Normal","Fixed Defect","Reversible Defect"][thal]]
    })
    st.dataframe(display, hide_index=True, use_container_width=True)

with col2:
    st.subheader("🔮 Prediction")
    if st.button("🔍 Predict Now", use_container_width=True, type="primary"):
        X_sc  = scaler.transform(input_df)
        proba = model.predict_proba(X_sc)[0][1]
        pred  = int(proba >= 0.5)

        if pred == 1:
            st.error("⚠️ Heart Disease Detected")
        else:
            st.success("✅ No Heart Disease Detected")

        st.metric("Risk Probability", f"{proba*100:.1f}%")
        risk = "🔴 HIGH" if proba > 0.7 else ("🟡 MEDIUM" if proba > 0.4 else "🟢 LOW")
        st.metric("Risk Level", risk)

        st.subheader("🧠 SHAP Explanation")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sc)
        fig, ax = plt.subplots(figsize=(8,4))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=X_sc[0],
                feature_names=feature_cols
            ), show=False
        )
        st.pyplot(plt.gcf(), use_container_width=True)
        plt.close()

st.markdown("---")
st.caption("⚠️ For educational purposes only. Not a substitute for medical advice.")
