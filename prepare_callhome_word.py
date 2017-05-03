# coding: utf-8

# ## Prepare parallel corpus
# 
# **Based on TensorFlow code: https://github.com/tensorflow/models/blob/master/tutorials/rnn/translate/data_utils.py**

# In[ ]:

import os
import re
import pickle
from tqdm import tqdm
import sys


# In[ ]:

from nmt_config import *


def create_vocab(text_fname):
    vocab = {}
    w2i = {}
    i2w = {}
    with open(text_fname,"rb") as in_f:
        for i, line in enumerate(in_f):
            word = line.strip()
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1
    
    print("vocab length: {0:d}".format(len(vocab)))
    
    vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    print("Vocab size={0:d}".format(len(vocab_list)))

    for i, w in enumerate(vocab_list):
        w2i[w] = i
        i2w[i] = w
            
    print("finished vocab processing for {0:s}".format(text_fname))
    
    return vocab, w2i, i2w


# In[ ]:

def create_input_config():
    en_name = os.path.join(input_dir, "vocab.en")
    fr_name = os.path.join(input_dir, "vocab.es")
    
    vocab_path = os.path.join(input_dir, "vocab.dict")
    w2i_path = os.path.join(input_dir, "w2i.dict")
    i2w_path = os.path.join(input_dir, "i2w.dict")
    
    # create vocabularies
    vocab = {"en":{}, "fr":{}}
    w2i = {"en":{}, "fr":{}}
    i2w = {"en":{}, "fr":{}}
    
    print("*"*50)
    print("en file")
    print("*"*50)
    vocab["en"], w2i["en"], i2w["en"] = create_vocab(en_name)
    print("*"*50)
    print("fr file")
    print("*"*50)
    vocab["fr"], w2i["fr"], i2w["fr"] = create_vocab(fr_name)
    print("*"*50)
    
    pickle.dump(vocab, open(vocab_path, "wb"))
    pickle.dump(w2i, open(w2i_path, "wb"))
    pickle.dump(i2w, open(i2w_path, "wb"))
    print("finished creating input config")

# In[ ]:

create_input_config()


# In[ ]:

