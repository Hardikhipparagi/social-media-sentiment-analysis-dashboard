import streamlit as st
import os
import joblib
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# ----------------------------
# Load Model
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "vectorizer.pkl"))

# ----------------------------
# UI HEADER
# ----------------------------
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

st.title("💬 Social Media Sentiment Analysis Dashboard")
st.markdown("Analyze sentiments from text or CSV files in real-time")

# ----------------------------
# OPTION SELECTOR
# ----------------------------
option = st.radio("Choose Input Type:", ["Single Text", "Upload CSV"])

# ============================
# 🔹 SINGLE TEXT MODE
# ============================
if option == "Single Text":
    text = st.text_area("Enter a comment:")

    if st.button("Analyze Sentiment"):
        if text.strip() == "":
            st.warning("Please enter text")
        else:
            vec = vectorizer.transform([text])
            pred = model.predict(vec)[0]

            if pred == "positive":
                st.success(f"😊 Sentiment: {pred}")
            elif pred == "negative":
                st.error(f"😠 Sentiment: {pred}")
            else:
                st.info(f"😐 Sentiment: {pred}")

# ============================
# 🔹 CSV MODE
# ============================
else:
    file = st.file_uploader("Upload CSV file (must have 'text' column)")

    if file is not None:
        df = pd.read_csv(file)

        if "text" not in df.columns:
            st.error("CSV must contain 'text' column")
        else:
            st.success("File uploaded successfully ✅")

            # Prediction
            df["sentiment"] = model.predict(vectorizer.transform(df["text"]))

            st.subheader("📊 Preview")
            st.dataframe(df.head())

            # ----------------------------
            # Sentiment Distribution
            # ----------------------------
            sentiment_counts = df["sentiment"].value_counts()

            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="Sentiment Distribution"
            )

            st.plotly_chart(fig)

            # ----------------------------
            # Metrics
            # ----------------------------
            total = len(df)
            pos = sentiment_counts.get("positive", 0)
            neg = sentiment_counts.get("negative", 0)
            neu = sentiment_counts.get("neutral", 0)

            col1, col2, col3 = st.columns(3)

            col1.metric("Positive 😊", f"{pos}", f"{(pos/total)*100:.1f}%")
            col2.metric("Negative 😠", f"{neg}", f"{(neg/total)*100:.1f}%")
            col3.metric("Neutral 😐", f"{neu}", f"{(neu/total)*100:.1f}%")

            # ----------------------------
            # Download Results
            # ----------------------------
            st.download_button(
                "📥 Download Results",
                df.to_csv(index=False),
                file_name="sentiment_results.csv"
            )