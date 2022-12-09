from load_data import load_data
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def create_dictionary(f):
    dictionary = {} 
    for doc in f:
        words = word_tokenize(doc[3])
        for word in words:
            word = WordNetLemmatizer().lemmatize(word)
            if word in dictionary:
                dictionary[word] += 1
            else:
                dictionary[word] = 1 

    return dictionary
            
        
def occurences(f, dictionary):
    lookup = dictionary.keys()
    inverse_lookup = {word: id for id, word in enumerate(lookup)}
    occurences = np.zeros((len(f), len(lookup)))
    for id_d, doc in enumerate(f):
        words = word_tokenize(doc[3])
        for word in words:
            word = WordNetLemmatizer().lemmatize(word)
            id_w = inverse_lookup[word]
            occurences[id_d][id_w] += 1 
    return occurences, inverse_lookup


def calcul_tf(f, occurences_matrix) :
    tf = np.zeros((len(occurences_matrix), len(occurences_matrix[0])))
    i = 0 
    for doc in f:
        words = word_tokenize(doc[3])
        nbwords = len(words)
        tf[i] = occurences_matrix[i] / nbwords
        i += 1   
    return tf

def calcul_idf(f, dictionary, occurences_matrix) :
    nbdocs = len(f)
    idf = np.full(len(dictionary), nbdocs)
    for j in range(0, len(occurences_matrix[0])):
        ite = 0
        for i in range(0, len(occurences_matrix)):
            if(occurences_matrix[i][j] > 0):
                ite += 1
        idf[j] = np.log(idf[j]/ite)
    return idf     
                
def tfidf(f):
    dictionary = create_dictionary(f)
    occurences_matrix, inverse_lookup = occurences(f, dictionary)
    tf = calcul_tf(f, occurences_matrix)
    idf = calcul_idf(f, dictionary, occurences_matrix)
    
    tfidf = np.zeros((len(tf), len(tf[0])))
    for i in range (0, len(tf)):
        for j in range(0, len(tf[0])):
            tfidf[i][j] = tf[i][j] * idf[j]
    
    return tfidf, tf, idf, inverse_lookup
    
    
          
