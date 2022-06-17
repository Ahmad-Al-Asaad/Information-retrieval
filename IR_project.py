from flask import Flask, request, jsonify
from collections import defaultdict
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import sys
import os
#import pandas as pd
#import numpy as np
import nltk.stem as stemmer
from collections import defaultdict
import re

from operator import itemgetter
import string
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller

app = Flask(__name__)

#app.config['SERVER_NAME'] = '0.0.0.0'

@app.route('/<string:query>/<int:dataset>')



def preparing_dataset(query: str, dataset: int):
    #return jsonify(message="My name is " + query + " and I am " + str(dataset) + " years old")
    ## Reading Datasets
    iDs_marker = re.compile('\.I ')

    def get_data(path, marker):
        with open(path, 'r') as f:
            text = f.read().replace('\n', ' ')
            lines = re.split(marker, text)
            lines.pop(0)
        return lines

    # choosing Dataset According to Request
    if (dataset == 1):
        corpus_data = get_data('/Users/Ahmad98/Desktop/IR_proj/CISI/CISI.all', iDs_marker)
        cisi_title_start = re.compile(' \.T')
        cisi_author_start = re.compile(' \.A')
        cisi_date_start = re.compile(' \.B')
        cisi_text_start = re.compile(' \.W')
        cisi_cross_start = re.compile(' \.X')

        # process the document data

        chunked_corpus = defaultdict(dict)

        for line in corpus_data:

            entries = re.split(cisi_title_start, line, 1)
            id = entries[0].strip()  # save the id
            no_id = entries[1]

            if len(re.split(cisi_author_start, no_id)) >= 2:  # is there just one author?
                no_id_entries = re.split(cisi_author_start, no_id, 1)
                chunked_corpus[id]['text'] = no_id_entries[0].strip()  # save title
                no_title = no_id_entries[1]

                if len(re.split(cisi_date_start, no_title)) > 1:  # is there a publication date?
                    no_title_entries = re.split(cisi_date_start, no_title)
                    chunked_corpus[id]['author'] = chunked_corpus[id]['text'] + ' ' + no_title_entries[
                        0].strip()  # save author
                    no_author = no_title_entries[1]
                    no_author_entries = re.split(cisi_text_start, no_author)
                    chunked_corpus[id]['publication_date'] = no_author_entries[0].strip()  # save publication date
                    no_author_date = no_author_entries[1]
                else:
                    no_title_entries = re.split(cisi_text_start, no_title)
                    chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + no_title_entries[
                        0].strip()  # save author
                    no_author_date = no_title_entries[1]

            else:
                no_id_entries = re.split(cisi_author_start, no_id)
                chunked_corpus[id]['text'] = no_id_entries[0].strip()  # save title
                chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + no_id_entries[
                    1].strip()  # save the first author
                no_title_entries = re.split(cisi_text_start, no_title)
                chunked_corpus[id]['text'] += chunked_corpus[id]['text'] + ' ' + no_title_entries[
                    0].strip()  # save the second author
                no_author_date = no_title_entries[1]

            last_entries = re.split(cisi_cross_start, no_author_date)
            chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + last_entries[0].strip()  # save text
            chunked_corpus[id]['cross-refrences'] = last_entries[1].strip()  # save cross references
    else:
        corpus_data = get_data('/Users/Ahmad98/Desktop/IR_proj/cacm/cacm.all', iDs_marker)
        corpus_chunk_title = re.compile('\.[T] ')
        corpus_chunk_txt = re.compile(' \.W ')
        corpus_chunk_txt_pub = re.compile(' \.[W,B] ')
        corpus_chunk_publication = re.compile(' \.[B] ')
        corpus_chunk_author = re.compile(' \.[A] ', re.MULTILINE)
        corpus_chunk_author_add_cross = re.compile(' \.[A,N,X] ', re.MULTILINE)
        # corpus_chunk_add_cross = re.compile(' \.[B,N,X] ')
        corpus_chunk_add_cross1 = re.compile(' \.[N,X] ')

        # process the document data
        chunked_corpus = defaultdict(dict)

        for line in corpus_data:
            entries = re.split(corpus_chunk_title, line)
            id = entries[0].strip()  # save id
            no_id = entries[1]

            if len(re.split(corpus_chunk_txt, no_id)) == 2:  # is there text?
                no_id_entries = re.split(corpus_chunk_txt_pub, no_id)
                chunked_corpus[id]['text'] = no_id_entries[0].strip()  # save title
                chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + no_id_entries[1].strip()  # save text
                no_title_txt = no_id_entries[2]

                if len(re.split(corpus_chunk_author, no_title_txt)) == 2:  # is there an author?
                    no_title_entries = re.split(corpus_chunk_author_add_cross, no_title_txt)
                    chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + no_title_entries[
                        0].strip()  # save publication date
                    chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + no_title_entries[
                        1].strip()  # save author
                    chunked_corpus[id]['add_date'] = no_title_entries[2].strip()  # save add date
                    chunked_corpus[id]['cross-references'] = no_title_entries[3].strip()  # save cross-references

                else:
                    no_title_entries = re.split(corpus_chunk_add_cross1, no_title_txt)
                    chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + no_title_entries[
                        0].strip()  # save publication date
                    chunked_corpus[id]['author'] = ''  # save author
                    chunked_corpus[id]['add_date'] = no_title_entries[1].strip()  # save add date
                    chunked_corpus[id]['cross-references'] = no_title_entries[2].strip()  # save cross-references

            else:
                no_id_entries = re.split(corpus_chunk_publication, no_id, 1)
                chunked_corpus[id]['text'] = no_id_entries[0].strip()  # save title
                no_title = no_id_entries[1]

                if len(re.split(corpus_chunk_author, no_title, 1)) == 2:  # is there an author?
                    no_title_entries = re.split(corpus_chunk_author_add_cross, no_title)
                    chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + no_title_entries[
                        0].strip()  # save publication date
                    chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + no_title_entries[
                        1].strip()  # save author
                    chunked_corpus[id]['add_date'] = no_title_entries[2].strip()  # save add date
                    chunked_corpus[id]['cross-references'] = no_title_entries[3].strip()  # save cross-references

                else:
                    no_title_entries = re.split(corpus_chunk_add_cross1, no_title)
                    chunked_corpus[id]['text'] = chunked_corpus[id]['text'] + ' ' + no_title_entries[
                        0].strip()  # save publication date
                    chunked_corpus[id]['author'] = ''  # save author
                    chunked_corpus[id]['add_date'] = no_title_entries[1].strip()  # save add date
                    chunked_corpus[id]['cross-references'] = no_title_entries[2].strip()  # save cross-references

    #else:
     #   return "there is no dataset"

    # Chunking Dataset To Dealing With It

    # Dictionary For National Symbols
    dictionary = {
        "u.s": "united states",
        "u.s.a": "united states",
        "u.n": "united nations",
        "i.e": "example",
        "e.g.": "for example",
        "m.p": "member of the house of lords",
        "IBM": "International Business Machines Corporation",
        "TSS": "Time Sharing System",
    }
    # changing Symbols To It's Own Known
    for store, x in chunked_corpus.items():
        text = x['text']
        for key in dictionary.keys():
            # print(dictionary[key])
            text = text.replace(key, dictionary[key])
            x['text'] = text

    # Normalizing Dataset
    corpus_dict = {}
    file = open("/Users/Ahmad98/Desktop/IR_Proj/cacm/common_words", "r")
    fileData = file.read()
    file.close()
    stopwords = re.findall("\S+", fileData)
    for store, x in chunked_corpus.items():
        tokens = word_tokenize(x['text'])
        tokens = [w.lower() for w in tokens]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # common_words = set(stopwords)
        words = [w for w in stripped if not w in stopwords]
        corpus_dict[store] = words

    # Stemming Dataset
    ps = PorterStemmer()
    stemm_list = []
    for store, x in chunked_corpus.items():
        for i in corpus_dict[store]:
            stemm_list.append(ps.stem(i))
        corpus_dict[store] = stemm_list
        stemm_list = []

    # Lemmatizeing Dataset
    lemmatizer = WordNetLemmatizer()
    lemmatize_list = []
    for store, x in chunked_corpus.items():
        for i in corpus_dict[store]:
            lemmatize_list.append(lemmatizer.lemmatize(i))
        corpus_dict[store] = lemmatize_list
        lemmatize_list = []


    ###############################

    #####query Processing#########

    ###############################
    def expand(text):
        for key in dictionary.keys():
            text = text.replace(key, dictionary[key])
        return text

    qur = expand(query)

    # Normalizing Query1
    def remove_common_words_tok_lower(query):
        filtered = []
        for w in word_tokenize(query.lower()):
            if w not in stopwords:
                filtered.append(w)
        return filtered

    qur = remove_common_words_tok_lower(qur)

    # Normalizing Query2
    def remove_punctuation(data):
        symbols = "!\"#$%&()*+-/:;<=>?@[\]^_`{|}~,\n"
        ls = []
        for i in data:
            if i not in symbols:
                ls.append(i)
        return ls

    qur = remove_punctuation(qur)

    # lemmatization Query
    lemmatizer = WordNetLemmatizer()

    def lemmtization_query(query):
        lemmatized_string = [lemmatizer.lemmatize(words) for words in query]
        return lemmatized_string

    qur = lemmtization_query(qur)

    # Stemming Query
    ps = PorterStemmer()

    def stemming_query(query):
        stemm = [ps.stem(words) for words in query]
        return stemm

    qur = stemming_query(qur)

    # Put All Tokenized Word In Query Togather
    query_text = []
    string1 = ''
    for i in qur:
        string1 = string1 + str(i) + ' '
    query_text.append(string1)

    # Put All Tokenized Word In Dataset Togather
    txt_doc = ''
    txt_index = []
    for store, x in corpus_dict.items():
        for i in x:
            txt_doc = txt_doc + i + ' '
        txt_index.append(txt_doc)
        txt_doc = ''
    print(txt_index)

    # TF-iDF Calculating
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(txt_index)
    tfidf_matrix_test = tfidf_vectorizer.transform(query_text)

    # Cosine Similarty
    cos = cosine_similarity(tfidf_matrix_train, tfidf_matrix_test)

    # Getting Top 10 Docs From cosine Function
    s = {}
    h = 1
    for i in cos:
        for j in i:
            s[h] = j
        h = h + 1
    res = dict(sorted(s.items(), key=itemgetter(1), reverse=True)[:10])

    # def get_relevent_docs(res):
    #     ls = []
    #     for v, k in res.items():
    #         ls.append(txt_index[v])
    #     return ls
    ls = []
    for v, k in res.items():
        for i, j in chunked_corpus.items():
            if (int(v) == int(i)):
                text = j
                ls.append(text)

   # docs=get_relevent_docs(res)
    return jsonify(ls)

if __name__ == '__main__':
    #app.run()
    app.run(host='0.0.0.0')
