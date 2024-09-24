from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120), nullable=False)
    picture = db.Column(db.String(200), nullable=True)
    auth0_id = db.Column(db.String(120), unique=True, nullable=False)

with app.app_context():
    db.create_all()

model_path = "./fine_tuned_distilbert2"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

if hasattr(model.config, 'id2label'):
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
else:
    labels = ["negative", "positive", "neutral"]

# Initialize Spacy's NER model
nlp = spacy.load("en_core_web_sm")

# Predict sentiment for a given text
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
    predicted_index = torch.argmax(outputs.logits).item()
    
    if predicted_index >= len(labels):
        # Handle the case where the predicted index is out of range
        return "unknown", 0.0
    
    predicted_label = labels[predicted_index]
    confidence = max(scores)
    return predicted_label, confidence

# Analyze text by breaking it into paragraphs
def analyze_text(text):
    paragraphs = text.split("\n\n")  # Split text by paragraphs
    analysis_results = []

    for paragraph in paragraphs:
        if paragraph.strip():  # Ignore empty paragraphs
            sentiment, confidence = predict_sentiment(paragraph)
            analysis_results.append({
                "paragraph": paragraph,
                "sentiment": sentiment,
                "confidence": round(confidence, 2)
            })
    
    return analysis_results

# Key Phrase Extraction using TF-IDF
def extract_key_phrases(reviews):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))  # Unigrams and bigrams
    X = vectorizer.fit_transform(reviews)
    feature_names = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    phrases = dict(zip(feature_names, scores))
    sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
    return sorted_phrases[:10]  # Top 10 key phrases

# Named Entity Recognition (NER)
def extract_entities(reviews):
    entities = []
    for review in reviews:
        doc = nlp(review)
        entities.extend([(ent.text, ent.label_) for ent in doc.ents])
    return entities

# Scrape Amazon reviews
def scrape_amazon_reviews(url, max_reviews=100):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Navigate to the product page first
        driver.get(url)
        time.sleep(2)  # Wait for the page to load

        # Find and click the "See more reviews" link
        try:
            see_all_reviews_link = driver.find_element(By.XPATH, "//a[@data-hook='see-all-reviews-link-foot']")
            see_all_reviews_link.click()
            time.sleep(2)  # Wait for the review page to load
        except Exception as e:
            return {"error": f"Could not find 'See more reviews' link: {str(e)}"}

        reviews = []
        current_page = 1
        
        while len(reviews) < max_reviews:
            # Get all review elements on the current page
            review_elements = driver.find_elements(By.XPATH, "//span[@data-hook='review-body']")
            
            # Extract review text and add to the list
            for review_element in review_elements:
                reviews.append(review_element.text)
                if len(reviews) >= max_reviews:
                    break  # Stop when we have enough reviews
            
            # Check if a "Next" button is available for pagination
            try:
                next_button = driver.find_element(By.XPATH, "//li[@class='a-last']/a")
                if 'a-disabled' in next_button.get_attribute('class'):
                    break  # No more pages, stop the loop
                next_button.click()
                time.sleep(2)  # Wait for the new page to load
                current_page += 1
            except Exception:
                # No "Next" button found, or it's disabled
                break
        
    except Exception as e:
        return {"error": f"Error occurred while scraping: {str(e)}"}
    finally:
        driver.quit()

    return reviews[:max_reviews]  # Return at most `max_reviews`

# API route to scrape Amazon reviews and analyze sentiment
@app.route("/scrape_reviews", methods=["POST"])
def scrape_and_analyze_reviews():
    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Scrape reviews from the given URL
    reviews = scrape_amazon_reviews(url)
    if isinstance(reviews, dict):  # If an error occurred during scraping
        return jsonify(reviews), 500

    # Analyze sentiment for each review
    analysis_results = []
    print("analyzing sentiment...") 
    for review in reviews:
        sentiment, confidence = predict_sentiment(review)
        analysis_results.append({
            "review": review,
            "sentiment": sentiment,
            "confidence": round(confidence, 2)
        })

    # Extract key phrases and entities
    print("Extracting key phrases...")  # Log before extraction
    key_phrases = extract_key_phrases(reviews)
    entities = extract_entities(reviews)

    return jsonify({
        "url": url,
        "reviews_analysis": analysis_results,
        "key_phrases": key_phrases,
        "entities": entities
    })

# Route to store user data
@app.route("/users", methods=["POST"])
def store_user():
    data = request.json

    email = data.get('email')
    name = data.get('name')
    picture = data.get('picture')
    auth0_id = data.get('auth0Id')

    if not email or not auth0_id:
        return jsonify({"error": "Missing required user data"}), 400

    # Check if user already exists
    existing_user = User.query.filter_by(auth0_id=auth0_id).first()
    
    if existing_user:
        return jsonify({"message": "User already exists"}), 200

    # Create a new user
    new_user = User(email=email, name=name, picture=picture, auth0_id=auth0_id)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User stored successfully"}), 201

if __name__ == '__main__':
    app.run(debug=True)
