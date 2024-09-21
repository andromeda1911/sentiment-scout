import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
import matplotlib.pyplot as plt

# Load the fine-tuned sentiment analysis model and tokenizer
model_path = "./fine_tuned_distilbert1"  # Replace with your model directory
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define the sentiment labels
labels = ["negative", "positive"]

# Function to predict sentiment
def predict_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        scores = torch.softmax(outputs.logits, dim=1).tolist()[0]
        predicted_index = torch.argmax(outputs.logits).item()

        if predicted_index < len(labels):
            predicted_label = labels[predicted_index]
            confidence = max(scores)
        else:
            predicted_label = "unknown"
            confidence = 0.0

        return predicted_label, confidence

    except Exception as e:
        st.error(f"Error in sentiment prediction: {str(e)}")
        return "unknown", 0.0

# Function to scrape Amazon reviews
def scrape_amazon_reviews(url, max_reviews=100):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    reviews = []
    try:
        driver.get(url)
        time.sleep(2)

        try:
            see_all_reviews_link = driver.find_element(By.XPATH, "//a[@data-hook='see-all-reviews-link-foot']")
            see_all_reviews_link.click()
            time.sleep(2)
        except Exception as e:
            return {"error": f"Could not find 'See more reviews' link: {str(e)}"}

        while len(reviews) < max_reviews:
            review_elements = driver.find_elements(By.XPATH, "//span[@data-hook='review-body']")
            for review_element in review_elements:
                reviews.append(review_element.text)
                if len(reviews) >= max_reviews:
                    break

            try:
                next_button = driver.find_element(By.XPATH, "//li[@class='a-last']/a")
                if 'a-disabled' in next_button.get_attribute('class'):
                    break
                next_button.click()
                time.sleep(2)
            except Exception:
                break
    finally:
        driver.quit()

    return reviews[:max_reviews]

# Function to extract topics using LDA
def extract_topics(reviews, n_topics=3):
    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(reviews)

    lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda_model.fit(dtm)

    topics = []
    for index, topic in enumerate(lda_model.components_):
        topic_terms = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-5:]]  # Top 5 words
        topics.append(f"Topic {index + 1}: {', '.join(topic_terms)}")

    return topics

# Function to create a heatmap visualization
def create_sentiment_heatmap(sentiments, reviews):
    sentiment_data = {
        'Sentiment': sentiments,
        'Reviews': reviews
    }
    plt.figure(figsize=(10, 5))
    sns.heatmap([[sentiments.count("positive"), sentiments.count("negative")]], annot=True, cmap="coolwarm", fmt="d")
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Heatmap')
    st.pyplot(plt)

# Streamlit App
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: white;
    }
    .stButton>button {
        color: white;
        background-color: #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Landing Page
st.title("🌟 Sentiment Scout 🌟")
st.write("""
### Analyze customer reviews and get valuable insights with our powerful sentiment analysis and topic extraction tool. 
Simply paste the Amazon product URL and let us do the magic!
""")

# Dark-themed description
st.markdown("""
    This application allows you to:
    - 📝 **Scrape and analyze reviews** directly from Amazon product pages.
    - 🎯 **Predict sentiment** for each review using advanced machine learning models.
    - 🧠 **Extract common themes and topics** from customer feedback.
    - 📊 Visualize the sentiment distribution with **heatmaps** for quick insights.
""")

# Search Bar
url = st.text_input("🔍 Enter the Amazon product review URL:")

# Button to start scraping and analysis
if url:
    # Show skeleton loader during review scraping
    with st.spinner("Scraping reviews and analyzing..."):
        placeholder = st.empty()  # Placeholder to show skeleton

        time.sleep(2)  # Simulate loading time

        # Scrape the reviews
        reviews = scrape_amazon_reviews(url)

        # If an error occurs, show the error message
        if isinstance(reviews, dict):
            st.error(reviews.get("error"))
        else:
            placeholder.empty()  # Remove skeleton loader

            st.success(f"Scraped {len(reviews)} reviews successfully!")

            # Analyze sentiment for each review
            st.write("Analyzing sentiment...")
            analysis_results = []
            sentiments = []
            for review in reviews:
                sentiment, confidence = predict_sentiment(review)
                analysis_results.append({
                    "review": review,
                    "sentiment": sentiment,
                    "confidence": round(confidence, 2)
                })
                sentiments.append(sentiment)

            # Display sentiment results
            for result in analysis_results:
                st.write(f"Review: {result['review']}")
                st.write(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']})")
                st.write("---")

            # Display Heatmap for sentiments
            create_sentiment_heatmap(sentiments, reviews)

            # Extract common themes
            st.write("Extracting common themes...")
            common_themes = extract_topics(reviews)
            for theme in common_themes:
                st.write(theme)

# Footer
st.write("---")
st.markdown("""
    Made with ❤️ by **gemnikodes**.
""")
