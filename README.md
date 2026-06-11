# NLP Sentiment Analysis Web App

An end-to-end Natural Language Processing (NLP) application that predicts the sentiment of user-entered product reviews using a machine learning model.

## 🚀 Features
* **Machine Learning Pipeline:** Text preprocessing (lowercasing, punctuation removal, tokenization, and stopword filtering) utilizing NLTK.
* **TF-IDF Vectorization:** Converts cleaned text into numerical features for the model.
* **XGBoost Classifier:** A powerful gradient-boosting model trained to classify reviews as Positive or Negative.
* **Interactive UI:** A clean, responsive web interface built entirely with Streamlit.

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Libraries:** Streamlit, XGBoost, Scikit-learn, Pandas, NLTK
* **Environment Manager:** uv

## 📦 How to Run Locally
1. Clone this repository.
2. Install the required dependencies from `requirements.txt`.
3. Train the model to generate the pickle files:
   ```bash
   python train_model.py
