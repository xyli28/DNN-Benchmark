#!/usr/bin/env python3.6
from sys import argv
import pandas as pd
import numpy as np
import csv
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, concatenate
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

hidden_size = 64
embed_size = 128
batch_size = 10    
vocab_sizes = 30000   #cutoff for num of most common words for text
max_len = 300          #cutoff for length of sequence for text



def modelGen(max_len, vocab_sizes):
    text_input = Input(shape=(max_len,), name='text_input')
    text_embedding = Embedding(output_dim=embed_size, input_dim=vocab_sizes, 
                     input_length=max_len)(text_input)
    text_lstm = lstm_out = LSTM(hidden_size, dropout = 0.2, recurrent_dropout = 0.2)(text_embedding)
    output = Dense(1, activation='sigmoid', name='text_output')(text_lstm)
 
    model = Model(inputs=[text_input], outputs=[output]) 

    #adam = Adam(lr=0.01)
    model.compile(optimizer="adam", loss='binary_crossentropy', 
                  metrics = ['binary_accuracy']) 
    
    return model

def main():
 
    #Read csv data into a dataframe 
    data = pd.read_csv("Reviews.csv").sample(10000)
    #data = pd.read_csv("Reviews.csv").sample(frac=1)
    split = round(data.shape[0]*0.8)
    print (split)    
   
    #Convert training text to sequence
    t = Tokenizer(num_words=vocab_sizes)
    t.fit_on_texts(data['Text'])
    x = t.texts_to_sequences(data['Text'].astype(str))
    #Pad the training sequence length to max_len
    x = sequence.pad_sequences(x, maxlen=max_len)
    

    #Encode the ordinal label for example
    y = data['Score'].apply(lambda x: int(x >= 4)).to_numpy()
     
    model = modelGen(max_len, vocab_sizes) 
    hist = model.fit([x[:split]], [y[:split]], epochs=10, batch_size=batch_size,
           validation_data=([x[split:]], [y[split:]]))
 
if __name__ == "__main__":
    #np.random.seed(0)
    main()
