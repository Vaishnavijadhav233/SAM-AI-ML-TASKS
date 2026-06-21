import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import fetch_20newsgroups

# Download dataset - 20newsgroups वापरतोय, त्यात pos/neg/neutral सारखे topics आहेत
print("Dataset loading...")
categories = ['rec.autos', 'talk.politics.misc', 'sci.space'] # 3 sentiment सारखे
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

df = pd.DataFrame({'text': data.data, 'sentiment': data.target})
df = df[df['text'].str.len() > 10] # empty reviews काढून टाक

# Text cleaning
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

df['clean_text'] = df['text'].apply(clean_text)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['sentiment'], test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model - Naive Bayes
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Prediction + Accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=categories))

# Test with your own review
sample_review = "This product is amazing and works perfectly!"
sample_clean = clean_text(sample_review)
sample_vec = vectorizer.transform([sample_clean])
pred = model.predict(sample_vec)
print(f"\nSample Review: '{sample_review}'")
print(f"Predicted Sentiment: {categories[pred[0]]}")
