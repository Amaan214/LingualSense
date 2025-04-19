import io
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from joblib import load
import numpy as np
import re
from langdetect import detect, DetectorFactory
import plotly.express as px
import pandas as pd

# To ensure consistent language detection
DetectorFactory.seed = 0

# --------------------------------------Text Preprocessing-----------------------------------------
def clean_text(text):
    text = re.sub(r'\\', '', text)  # remove backslashes
    text = re.sub(r'[^\w\s.,?!-]', '', text)  # remove special characters
    text = re.sub(r'\s+', ' ', text)  # remove multiple spaces with single space
    text = re.sub(r'\d+', '', text)  # remove any numbers
    text = text.lower()
    return text

# ------------------------------------------Load Model--------------------------------------
def load_resources(model_type = "GRU"):
    try:
        tokenizer = load('tokenizer.joblib')
        label_encoding = load('label_encoder.joblib')
        model_select = 'gru_model.h5' if model_type == "GRU" else 'lstm_model.h5'
        model = tf.keras.models.load_model(model_select)
        return tokenizer, label_encoding, model
    except FileNotFoundError as e:
        st.error(f"Error loading resources: {e}")
        st.stop()

# -------------------------------------Sentiment analysis--------------------------------------------
def sentiment_analysis(texts, tokenizer, encoder, model):
    language_list = []  # List to store detected languages and sentiments

    for text in texts:
        try:
            language_detected = detect(text)  # Detect the language of the text
        except:
            language_detected = "unknown"  # In case language detection fails
    
        # Clean the text for sentiment analysis
        process_text = clean_text(text)
        sequences = tokenizer.texts_to_sequences([process_text])
        padded_sequence = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    
        # Predict sentiment
        prediction = model.predict(padded_sequence, verbose=0)
        predict_label_index = np.argmax(prediction)
        predict_class_label = encoder.inverse_transform([predict_label_index])[0]
        
        # Append the results as a tuple
        language_list.append((text, language_detected, predict_class_label))

    return language_list

# --------------------------------------Split sentence----------------------------------
def split_sentence(text):
    # Split based on punctuation followed by whitespace OR capital letter (for missing space)
    rough_sentences = re.split(r'(?<=[.!?।؟！？。｡⸮])(?=\s+|[A-Z])', text)
    # Clean up extra whitespace and short fragments
    sentences = [s.strip() for s in rough_sentences if len(s.strip()) > 2]
    return sentences


# --------------------------------------Streamlit App-------------------------------------------------------------------
st.set_page_config(page_title="LingualSense: Multilingual Sentiment Analyzer", layout="wide")
st.title("🌐 LingualSense - Multilingual Sentiment Analyzer")

user_input = st.text_area("✍️ Enter text for sentiment analysis:", height=150)

# Upload option below input
uploaded_file = st.file_uploader("📄 Or upload a text file instead", type=["txt"])
if uploaded_file is not None:
    user_input = uploaded_file.read().decode("utf-8")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
model_type = st.sidebar.selectbox("Choose Model", ["LSTM", "GRU"])
show_chart = st.sidebar.checkbox("Show Sentiment Visualization", value=True)
enable_download = st.sidebar.checkbox("Click to download csv file", value=True)

if st.button("🔍 Analyze"):
    if user_input:
        with st.spinner("Analyzing..."):
            user_texts = split_sentence(user_input)
            print(user_input)
            tokenizer, encoder, model = load_resources(model_type)
            results = sentiment_analysis(user_texts, tokenizer, encoder, model)
            df_results = pd.DataFrame(results, columns=["Sentence", "Language", "Sentiment"])

            st.subheader("📊 Summary")
            sentiment_summary = df_results['Sentiment'].value_counts().reset_index()
            sentiment_summary.columns = ['Sentiment', 'Count']
            st.dataframe(sentiment_summary)

            st.subheader("📄 Detailed Analysis")
            for i, row in df_results.iterrows():
                with st.expander(f"Sentence {i+1}"):
                    st.markdown(f"**Sentence:** {row['Sentence']}")
                    st.markdown(f"**Language:** {row['Language'].capitalize()}")
                    st.markdown(f"**Sentiment:** {row['Sentiment']}")

            # Visualization
            if show_chart:
                st.subheader("📈 Sentiment Distribution")
                sentiment_percent = df_results['Sentiment'].value_counts(normalize=True).reset_index()
                sentiment_percent.columns = ['Sentiment', 'Percentage']
                sentiment_percent['Percentage'] *= 100

                fig = px.bar(sentiment_percent,
                            x='Sentiment', y='Percentage',
                            color='Sentiment', text=sentiment_percent['Percentage'].round(2),
                            color_discrete_sequence=px.colors.qualitative.Vivid,
                            title="Sentiment Distribution")
                fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig.update_layout(yaxis_range=[0, 100])
                st.plotly_chart(fig, use_container_width=True)

            # CSV Download
            if enable_download:
                csv_buffer = io.StringIO()
                df_results.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📥 Download results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )
    else:
        st.warning("Please enter some text to analyze.")

