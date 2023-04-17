# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 20:57:41 2023

@author: Dell
"""
import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the model
model = load_model('suicide_model.h5')


st.title('Mental Health App')

# Define a function to preprocess the input text
def preprocess_text(text):
    # Tokenize the input text
    sequence = tokenizer.texts_to_sequences([text])
    
    # Pad the sequence to the same length as the training data
    sequence = pad_sequences(sequence, maxlen=7796)
    
    return sequence


# Define a function to make predictions
def predict(user_input):
    # Preprocess the input text
    sequence = preprocess_text(user_input)
    
    # Make a prediction using the LSTM model
    prediction = model.predict(sequence)
    
    if prediction[0][0] > 0.5:
        return "The Text Contains References to self-harm"
    else:
        return "non suicidal"
    
# Create a text input for the user to enter their text
text = st.text_area('Enter your text here:')

# Make a prediction when the user clicks the button
if st.button('Predict'):
    prediction = predict(text)
    st.title(prediction)


