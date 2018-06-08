from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import nltk
import pandas as pd

json_file = open('Model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("Model/model.h5")

vocab = pd.read_csv('vocab.txt',sep=" ", header=None)
vocab = vocab[0].values
vocab = np.append(vocab,"UNKNOWN")
vocab = np.append(vocab,"ENDPAD")

word_map = {}
for index,value in enumerate(vocab):
    word_map[value] = index
n_words = vocab.shape[0]

def get_matrix_ids(s):
    id_matrix = []
    w = nltk.word_tokenize(s)
    w = [i.lower() for i in w if i.isalpha()]
    
    for i in w:
        if i in vocab:
            id_matrix.append(word_map[i])
        else :
            id_matrix.append(word_map["UNKNOWN"])
    return id_matrix

def rate_sentence(s):
    w = get_matrix_ids(s)
    w = pad_sequences(maxlen=65, sequences=[w], padding="post", value=n_words - 1)
    output = loaded_model.predict([w])[0]
    output = np.argmax(output)
    print("Predicted rating : " + str(output))


sentence = input()
rate_sentence(sentence)