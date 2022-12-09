from load_data import load_data
from tfidf import *
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def create_query_dictionary(query):
    dictionary = {} 
    words = word_tokenize(query)
    for word in words:
        word = WordNetLemmatizer().lemmatize(word)
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1 
    return dictionary

def best_results(scores, nb_result):
    scores_indexes = []
    doc_indexes = []
    for i in range(len(scores)):
        scores_indexes.append((scores[i], i+1))
    scores_indexes.sort(reverse=True)
    result = scores_indexes[:nb_result]
    for pair in result:
        doc_indexes.append(pair[1])
    return result, doc_indexes

def avgdl_calcul(all_file):
    avgdl = 0
    for doc in all_file:
        word_list = doc[3].split()
        avgdl += len(word_list)
    return avgdl / len(all_file)

def test_queries_bm25(all_file, qry_file, tf_matrix, idf_matrix,inverse_lookup, nb_results, avgdl, k, b):
    all_scores = []
    all_indexes = []
    for q in qry_file:
        query = q[1]
        words = create_query_dictionary(query).keys()
        scores = np.zeros(len(tf_matrix))
        for i in range (0, len(tf_matrix)):
            for word in words:
                if(word in inverse_lookup):
                    word_i = inverse_lookup[word]
                    scores[i] += idf_matrix[word_i] * (tf_matrix[i][word_i] * (k + 1)) / (tf_matrix[i][word_i] + k * (1 -b + b * (len(all_file[i][3].split()) / avgdl)))
        result, indexes = best_results(scores, nb_results)
        all_scores.append(result)
        all_indexes.append(indexes)
    return all_scores, all_indexes

def test_queries_tfidf(qry_file, tfidf_matrix, inverse_lookup, nb_results):
    all_scores = []
    all_indexes = []
    for q in qry_file:
        query = q[1]
        words = create_query_dictionary(query).keys()
        scores = np.zeros(len(tfidf_matrix))
        for i in range (0, len(tfidf_matrix)):
            for word in words:
                if(word in inverse_lookup):
                    word_i = inverse_lookup[word]
                    scores[i] += tfidf_matrix[i][word_i]
        result, indexes = best_results(scores, nb_results)
        all_scores.append(result)
        all_indexes.append(indexes)
    return all_scores, all_indexes
    

def evaluation_scores(indexes, rel_file):
    precision = 0.
    rappel = 0.
    for key, docs in rel_file.items():
        vp = 0.
        fp = 0.
        fn = 0.
        for doc in docs:
            if(doc in indexes[key-1]):
                vp+=1
            else:
                fn+=1
        fp += len(indexes[key-1]) - vp

        precision += vp / (vp + fp)
        rappel += vp / (vp + fn)
        
    precision /= len(rel_file)
    rappel /= len(rel_file)
    f_score = 2 * (precision * rappel) / (precision + rappel)
    
    return precision, rappel, f_score



all_file, qry_file, rel_file = load_data()

avgdl = avgdl_calcul(all_file)

tfidf_matrix, tf_matrix, idf_matrix, inverse_lookup = tfidf(all_file)

nb_results = 10
k = 1.2
b = 0.75

scores_tfidf, indexes_tfidf = test_queries_tfidf(qry_file, tfidf_matrix, inverse_lookup, nb_results)
scores_bm25, indexes_bm25 = test_queries_bm25(all_file, qry_file, tf_matrix, idf_matrix, inverse_lookup, nb_results, avgdl, k, b)

evaluation_tfidf = evaluation_scores(indexes_tfidf, rel_file)
evaluation_bm25 = evaluation_scores(indexes_bm25, rel_file)

print("Scores TFiDF : precision=", evaluation_tfidf[0], " rappel=", evaluation_tfidf[1], " f_score=", evaluation_tfidf[2])
print("Scores BM25 : precision=", evaluation_bm25[0], " rappel=", evaluation_bm25[1], " f_score=", evaluation_bm25[2])
