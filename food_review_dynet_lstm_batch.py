#!/usr/bin/env python3.6
from sys import argv
import pandas as pd
import numpy as np
import dynet as dy 
import csv
import time
from tensorflow.keras.preprocessing.text import Tokenizer


hidden_size = 64
embed_size = 128
batch_size = 10       
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
    data = pd.read_csv("Reviews.csv").sample(10000)
    #data = pd.read_csv("Reviews.csv").sample(frac=1)
    split = round(data.shape[0]*0.8)
    split_2 = len(data)-split
    print (split)    
   
    #Set up the training parameters
    vocab_sizes = 30000   #cutoff for num of most common words for text
    #max_len = 300          #cutoff for length of sequence for text

    #Convert training text to sequence
    t = Tokenizer(num_words=vocab_size)
    t.fit_on_texts(data['Text'])
    x = t.texts_to_sequences(data['Text'].astype(str))
    y = data['Score'].apply(lambda x: int(x >= 4))

    #z = list(zip(x,y))
    #z.sort(key = lambda s : len(s[0]))
    #x = [s[0] for s in z]
    #y = [s[1] for s in z]

    m = dy.Model()
    trainer = dy.AdamTrainer(m)
    embeds = m.add_lookup_parameters((vocab_size, embed_size))
    acceptor = LstmAcceptor(embed_size, hidden_size, 1, m) 

    sum_of_losses = 0.0
    for epoch in range(10):
        start = time.time()
        for k in range(split//batch_size):
            dy.renew_cg() 
            losses = []
            for sequence, label in zip(x[k*batch_size:(k+1)*batch_size], y[k*batch_size:(k+1)*batch_size]):
                label = dy.scalarInput(label)
                vecs = [embeds[i] for i in sequence]
                preds = acceptor(vecs)
                loss = dy.binary_log_loss(preds, label)
                losses.append(loss)
            batch_loss = dy.esum(losses)/batch_size
            sum_of_losses += batch_loss.value()
            batch_loss.backward()
            trainer.update()
        print ("train loss %f time %f sentences %d" %(sum_of_losses/(split//batch_size), time.time()-start, split))
        sum_of_losses = 0.0

        correct = 0
        for k in range(split_2//batch_size):
            dy.renew_cg() 
            for sequence, label in zip(x[split+k*batch_size:split+(k+1)*batch_size], y[split+k*batch_size:split+(k+1)*batch_size]):
                label = dy.scalarInput(label)
                vecs = [embeds[i] for i in sequence]
                preds = acceptor(vecs)
                correct += int(preds.value()) == int(label.value())
                sum_of_losses += dy.binary_log_loss(preds, label).value()
        print ("test loss %f accuracy %f" %(sum_of_losses/split_2, correct/split_2))
        sum_of_losses = 0.0
 
if __name__ == "__main__":
    #np.random.seed(0)
    main()
