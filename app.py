import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained LSTM model
model = tf.keras.models.load_model("sentiment_lstm.h5")  # Ensure you have this file

# Load tokenizer (used for text preprocessing)
with open("tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

max_length = 100  # Ensure this matches what was used during training

# Streamlit App
st.title("Movie Review Sentiment Analysis")

review = st.text_area("Enter your movie review:")
if st.button("Predict Sentiment"):
    if review:
        # Preprocess the input review
        review_seq = tokenizer.texts_to_sequences([review])
        review_pad = pad_sequences(review_seq, padding='post', maxlen=max_length)
        
        # Predict sentiment
        prediction = model.predict(review_pad)[0][0]  # Extract scalar value
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        
        st.write(f"**Sentiment:** {sentiment}")
    else:
        st.warning("Please enter a review before predicting.")
