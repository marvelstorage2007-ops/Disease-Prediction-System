
import streamlit as st
import pandas as pd
import joblib

# Page settings
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Load trained model and supporting files
model = joblib.load("random_forest_model.pkl")
encoder = joblib.load("label_encoder.pkl")
symptom_columns = joblib.load("symptom_columns.pkl")

# Dashboard heading
st.title("Disease Prediction System")
st.write("A machine learning based system for symptom-based disease prediction.")

st.info("This system is for educational purposes only.")

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:
    st.metric("ML Algorithm", "Random Forest")

with col2:
    st.metric("Available Diseases", len(encoder.classes_))

st.success("Trained model loaded successfully!")
