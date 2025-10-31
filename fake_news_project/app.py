from flask import Flask, render_template, request
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# نموذج بسيط ومكتبة TF-IDF
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-zA-Z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# بيانات بسيطة للتجريب
texts_real = [
    "NASA confirms water detected on Mars surface.",
    "Ministry of Health announces new vaccination campaign.",
]
texts_fake = [
    "Aliens have landed in Egypt and took over the pyramids.",
    "Drinking bleach cures all diseases including COVID-19.",
]

data = texts_real + texts_fake
labels = [0]*len(texts_real) + [1]*len(texts_fake)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([clean_text(x) for x in data])
model = LogisticRegression()
model.fit(X, labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    cleaned = clean_text(text)
    X_test = vectorizer.transform([cleaned])
    pred = model.predict(X_test)[0]
    prob = model.predict_proba(X_test)[0][1]
    label = "Fake ❌" if pred == 1 else "Real ✅"
    return render_template('index.html', text=text, label=label, prob=round(prob, 3))

if __name__ == '__main__':
    app.run(debug=True)