import keras
import streamlit as st
from BanglaUtilities import bangla_text_preprocess
from keras.models import load_model
# import pandas as pd
# import numpy as np
# import pickle
# import re
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

print(keras.__version__)

st.write(''' # A Bangla news classifier Web App ''')
st.write("A web app that detects whether a news is Fake or Authentic")
article_input = st.text_input("Enter any Bangla news article...")
predict_button = st.button("Predict")

prepro_article = bangla_text_preprocess(article_input)

max_words = 15000 # considers only the top max_words number of words in the dataset
tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(prepro_article)

sequences = tokenizer.texts_to_sequences(prepro_article)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen = 150)

model0 = load_model('model0.h5', compile = False)
model1 = load_model('model1.h5', compile = False)
model2 = load_model('model2.h5', compile = False)
model3 = load_model('model3.h5', compile = False)


if article_input:
  pred0 = model0.predict(data)
  pred1 = model1.predict(data)
  pred2 = model2.predict(data)
  pred3 = model3.predict(data)

  real = 0
  fake = 0

  result = [pred0[0][0], pred1[0][0], pred2[0][0], pred3[0][0]]

  for pred_val in result:
    if pred_val < 0.5:
      fake += 1
    else:
      real +=1

  print(result, real, fake, pred0[0][0])

  if article_input != "":
    if real >= fake:
      st.write("Authentic News 😀")
    else:
      st.write("Fake News 😑")

