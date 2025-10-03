import re
import joblib
import nltk
import spacy
import contractions
from flask import Flask, request, jsonify

# Download NLTK resources (only first time)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load trained vectorizer and classifier
vectorizer = joblib.load("tfidf.pkl")
model = joblib.load("logistic_model.pkl")

# Flask app
app = Flask(__name__)

# --------- Preprocessing steps ---------
def filter_text(text):
    clean_text = re.sub(r"[^A-Za-z\s]", "", text)
    return clean_text.lower()

def fix_text(text):
    return contractions.fix(text)

def tokenize_text(text):
    doc = nlp(text)
    return [token.text for token in doc]

def lemmatize_text(tokens):
    doc = " ".join(tokens)
    tokens = nlp(doc)
    return [token.lemma_ for token in tokens]

def preprocess(text):
    text = filter_text(text)
    text = fix_text(text)
    tokens = tokenize_text(text)
    tokens = lemmatize_text(tokens)
    # 🚫 No stopword removal (to match training!)
    return " ".join(tokens)

# --------- API route ---------
@app.route("/predict", methods=["GET"])
def predict():
    text = request.args.get("text", default=None, type=str)
    if not text:
        return jsonify({"error": "Please provide ?text=your sentence"}), 400

    # Preprocess input
    processed = preprocess(text)

    # Vectorize
    X = vectorizer.transform([processed])

    # Predict
    prediction = int(model.predict(X)[0])
    
    # LogisticRegression supports probabilities
    probability = float(max(model.predict_proba(X)[0]))

    label_map = {0: "Tweet has negative sentiment", 1: "Tweet has positive sentiment"}
    sentiment = label_map.get(prediction, f"Label {prediction}")

    return jsonify({
        "text": text,
        "processed_text": processed,
        "prediction": sentiment,
        "confidence": probability
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
