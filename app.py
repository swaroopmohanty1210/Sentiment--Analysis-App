import streamlit as st # Used to create web application interface
import pickle # Loads the trained model and tf-idf vectorizer
import re # for text cleaning with regular expressions
import nltk # for preprocessing text

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# laod files
with open ("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

stop_words = set(stopwords.words("english"))

# Cleaning
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]',' ',text)

    words = word_tokenize(text)

    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# Prediction
def predict_sentiment(text):# Functions that predicts sentiment

    cleaned = clean_text(text)

    vector = tfidf.transform([cleaned]) # transforms cleaned text into numerical features

    prediction = model.predict(vector)

    return prediction[0]


# streamlit UI
st.set_page_config( page_title="Sentiment Analysis", layout="centered") # Page title

st.title("Sentiment Analysis App") # Browser tab title.

review = st.text_area("Enter Review")

if st.button("Predict"):

    if review:
        result = predict_sentiment(review)
        if result == 1:
            st.success("Positive Review ! Yay!")
        else:
            st.error("negative Review ! Ohh no")


    else:
        st.warning("Enter review text")
