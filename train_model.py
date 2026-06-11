import pandas as pd # for handling datasets and dataframes
import re # regular expression library: used for text cleaning.
import pickle # Used to save and load trained models.
import nltk # for text preprocessing

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer # Converts text data into numerical features.
from xgboost import XGBClassifier

nltk.download("punkt")
nltk.download("stopwords")


# Laod Dataset
data = pd.read_csv("dataset-P676-converted-1.csv")

#Merge columns
data['review'] = (data['title'].astype(str) + " " +data['body'].astype(str))

# Keep required columns
data = data[["review", "rating"]]

# stopwords
stop_words = set(stopwords.words("english"))

# Cleaning Function
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    words = word_tokenize(text)

    words = [ w for w in words if w not in stop_words ]

    return " ".join(words)


# Clean text
data["review"] = data["review"].apply(clean_text)

# Sentiment levels
def sentiment_label(rating):

    if rating >= 3:
        return "Positive"

    return "Negative"

data["sentiment"] = data["rating"].apply(sentiment_label)

# TFIDF 
tfidf = TfidfVectorizer(max_features = 2000)
X = tfidf.fit_transform(data["review"])
Y= data["sentiment"]

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state=42)


# Encoding

le = LabelEncoder()

Y_encoded = le.fit_transform(Y)


# Train Model
xgb = XGBClassifier(n_estimators = 100, learning_rate = 0.1, max_depth = 3)
xgb.fit(X ,Y_encoded)

# save TFIDF 
with open("tfidf_vectorizer.pkl", "wb") as f:# Creates binary file

    pickle.dump(tfidf, f) # saves TF-IDF vocabulary and settings

# File Created : tfidf_vectorizer.pkl

# save Model
with open( "xgb_model.pkl", "wb") as f: # Creates binary file

    pickle.dump(xgb, f) # stores trained XGB model

print("Model Saved succesully")

