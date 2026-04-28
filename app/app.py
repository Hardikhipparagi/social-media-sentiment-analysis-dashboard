import streamlit as st
import os
import joblib

st.title("Sentiment Analysis Dashboard")
st.write("App started ✅")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

st.write("Model path:", model_path)
st.write("Vectorizer path:", vectorizer_path)

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"ERROR: {e}")