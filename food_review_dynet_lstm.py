#!/usr/bin/env python3.6
from sys import argv
import pandas as pd
import numpy as np
import dynet as dy 
import csv
from tensorflow.keras.preprocessing.text import Tokenizer

hidden_size = 64
embed_size = 128
batch_size = 500       
vocab_size = 30000

class LstmAcceptor(object):
    def __init__(self, in_dim, lstm_dim, out_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self.W       = model.add_parameters((out_dim, lstm_dim))

    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        W = self.W.expr() # convert the parameter into an Expession (add it to graph)
        outputs = lstm.transduce(sequence)
        result = dy.logistic(W*outputs[-1])
        return result

def main():
 
    #Read csv data into a dataframe 
    data = pd.read_csv("Reviews.csv").sample(1000)
    #data = pd.read_csv("Reviews.csv").sample(frac=1)
    split = round(data.shape[0]*0.8)
    print (split)    
   
    #Set up the training parameters
    vocab_sizes = 30000   #cutoff for num of most common words for text
    #max_len = 300          #cutoff for length of sequence for text

    #Convert training text to sequence
    t = Tokenizer(num_words=vocab_size)
    t.fit_on_texts(data['Text'])
    x = t.texts_to_sequences(data['Text'].astype(str))
    y = data['Score'].apply(lambda x: int(x >= 4))
    
    m = dy.Model()
    trainer = dy.AdamTrainer(m)
    embeds = m.add_lookup_parameters((vocab_size, embed_size))
    acceptor = LstmAcceptor(embed_size, hidden_size, 1, m) 

    sum_of_losses = 0.0
    for epoch in range(10):
        for sequence, label in zip(x[:split], y[:split]):
            dy.renew_cg()
            label = dy.scalarInput(label)
            vecs = [embeds[i] for i in sequence]
            preds = acceptor(vecs)
            loss = dy.binary_log_loss(preds, label)
            sum_of_losses += loss.npvalue()
            loss.backward()
            trainer.update()
        print (sum_of_losses / split)
        sum_of_losses = 0.0
    print ("\n\nPrediction time!\n")
    for sequence, label in zip(x[split:], y[split:]):
        dy.renew_cg()
        vecs = [embeds[i] for i in sequence]
        preds = acceptor(vecs).value()
        print (preds, label)
 
if __name__ == "__main__":
    #np.random.seed(0)
    main()
