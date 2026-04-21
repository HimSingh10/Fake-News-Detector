import streamlit as st
import pickle
import os
import re
import nltk
from nltk.corpus import stopwords

st.title("Fake News Detector")

# Show files in folder (for debugging)
st.write("Files in folder:", os.listdir())

# Load model safely
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_news(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    decision = model.decision_function(vec)[0]

    confidence = abs(decision)

    label = "Fake News" if prediction == 0 else "Real News"
    return label, round(confidence, 2)

# UI
input_text = st.text_area("Enter News Text:")

if st.button("Predict"):
    if input_text.strip() != "":
        result, confidence = predict_news(input_text)
        st.success(result)
        st.info(f"Confidence Score: {confidence}")
    else:
        st.warning("Please enter some text!")