import streamlit as st
import pandas as pd
pip insatll plotly
import plotly.express as px
from wordcloud import WordCloud
import base64
from io import BytesIO
from textblob import TextBlob
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.set_page_config(layout="wide")

# Function to read uploaded file
def load_data(file):
    if file is not None:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
    return None

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

# Function to perform sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to generate wordcloud images
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    buffer = BytesIO()
    wordcloud.to_image().save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + img_str

# Function to get top words
def get_top_words(text, n=10):
    words = re.findall(r'\w+', text.lower())
    word_counts = Counter(words)
    return word_counts.most_common(n)

# File uploader
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Filter text columns
    text_columns = [col for col in df.columns if df[col].dtype == 'object']

    # Select text column for sentiment analysis
    text_column = st.selectbox("Select Text Column for Sentiment Analysis:", ["Select Text Data"] + text_columns)

    if text_column and text_column != "Select Text Data":
        # Preprocess text
        df['cleaned_text'] = df[text_column].apply(preprocess_text)

        # Perform sentiment analysis
        df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)

        # Streamlit app layout
        st.title("Textual Data Sentiment Analysis Dashboard")

        # Calculate KPIs
        positive_reviews = df[df['sentiment'] == 'Positive'].shape[0]
        negative_reviews = df[df['sentiment'] == 'Negative'].shape[0]
        neutral_reviews = df[df['sentiment'] == 'Neutral'].shape[0]

        # Display KPIs
        st.subheader("Key Performance Indicators (KPIs)")
        col1, col2, col3 = st.columns(3)
        col1.metric("Positive Reviews", positive_reviews)
        col2.metric("Negative Reviews", negative_reviews)
        col3.metric("Neutral Reviews", neutral_reviews)

        # Create columns for charts
        col1, col2 = st.columns(2)

        # Display sentiment analysis graph
        with col1:
            color_discrete_map = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'yellow'}
            fig = px.histogram(df, x='sentiment', title='Sentiment Analysis', color='sentiment', color_discrete_map=color_discrete_map)
            fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)
            st.plotly_chart(fig)

        # Display top words chart
        with col2:
            st.subheader("Top Words")
            all_text = " ".join(df['cleaned_text'])
            top_words = get_top_words(all_text)
            top_words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
            fig = px.bar(top_words_df, x='Word', y='Count', title='Top Words')
            st.plotly_chart(fig)

        # Display wordclouds
        positive_text = " ".join(df[df['sentiment'] == 'Positive']['cleaned_text'])
        negative_text = " ".join(df[df['sentiment'] == 'Negative']['cleaned_text'])
        positive_wordcloud = generate_wordcloud(positive_text)
        negative_wordcloud = generate_wordcloud(negative_text)

        col3, col4 = st.columns(2)
        with col3:
            st.image(positive_wordcloud, caption='Positive Wordcloud', use_container_width=True)
        with col4:
            st.image(negative_wordcloud, caption='Negative Wordcloud', use_container_width=True)

        # Display cleaned reviews table
        st.write("Cleaned Reviews Table")
        st.dataframe(df[[text_column, 'cleaned_text', 'sentiment']])
