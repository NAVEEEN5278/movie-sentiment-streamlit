import streamlit as st
import joblib
import re

# Load saved model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Clean input text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Streamlit UI
st.set_page_config(page_title="Movie Sentiment Analyzer")
st.title("🎬 Movie Review Sentiment Analyzer")

review = st.text_area("Enter your movie review below:")

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        st.success(f"Sentiment: **{sentiment}**")
