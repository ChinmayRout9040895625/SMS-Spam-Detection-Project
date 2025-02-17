import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Balanced dataset
corpus = [
    "Win a lottery, click here",
    "Hello, how are you?",
    "Get rich quickly with this scheme!",
    "This is a normal message",
    "Congratulations! You have won $1000",
    "Free money offer just for you",
    "Let's meet for lunch tomorrow",
    "Your order has been shipped",
    "Urgent! Your account has been compromised",
    "Hello, just checking in!"
]
labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]  # 1 = Spam, 0 = Not Spam

# Train the vectorizer
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus)

# Train model
model = MultinomialNB()
model.fit(X, labels)

# Save vectorizer and model
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model and vectorizer saved successfully.")




import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()


# Function to preprocess text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric words
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load vectorizer and model
with open("vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("📩 Email/SMS Spam Classifier")

# Input box
input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("⚠ Please enter a message to classify.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)

        # Vectorize (No `.fit_transform()`, only `.transform()`)
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)[0]

        # Display Result
        st.header("🚨 Spam" if result == 1 else "✅ Not Spam")



