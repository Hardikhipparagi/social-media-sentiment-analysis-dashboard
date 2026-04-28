import streamlit as st
import os
import joblib

# Title
st.title("💬 Social Media Sentiment Analysis Dashboard")
st.write("Analyze sentiment of comments in real-time")

# Load model safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# 🔹 USER INPUT
text = st.text_area("Enter a comment or tweet:")

# 🔹 PREDICTION
if st.button("Analyze Sentiment"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]

        # Display result
        if prediction == "positive":
            st.success(f"😊 Sentiment: {prediction}")
        elif prediction == "negative":
            st.error(f"😠 Sentiment: {prediction}")
        else:
            st.info(f"😐 Sentiment: {prediction}")