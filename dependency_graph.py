# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
from spacy.tokens import Doc
#nlp = StanfordCoreNLP('stanford-corenlp-4.2.0')
#tokenizer = lambda x: nlp.word_tokenize(x)

nlp = spacy.load('en_core_web_sm')
#nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim):
    embedding_matrix_file_name = '{0}_embedding_matrix.pkl'.format(str(embed_dim))
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = 'glove.6B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix
"""
def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        idx2graph[i] = adj_matrix
    pickle.dump(idx2graph, fout)
    fout.close()
"""
import os, json
import pandas as pd
import math
from tqdm import tqdm
  # for me this prints ['foo.json']
def process1(folder):
    json_files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
    max1 = 0
    k=0
    g_idx = 0
    global_word2idx = {}
    global_idx2word = {}
    for file in tqdm(json_files):
        text = []
        with open(folder+"/"+file) as jsonfile:
            data = json.load(jsonfile)
            l_idx = 0
            local_word2idx = {}
            local_idx2word = {}
            g2l = {}
            for i in range(len(data['nGrams'])):
                if len(data['nGrams'][i]['words']) == 1:
                    for j in range(len(data['nGrams'][i]['words'])):
                        if len(nlp(data['nGrams'][i]['words'][j]['text'])) ==1:
                            word = nlp(data['nGrams'][i]['words'][j]['text']).text.lower()
                            data['nGrams'][i]['words'][j]['nlp'] = word
                            text.append(data['nGrams'][i]['words'][j])
                            if word not in global_word2idx:
                                global_word2idx[word] = g_idx
                                global_word2idx[g_idx] = word
                                g_idx += 1
                            if word not in local_word2idx:
                                local_word2idx[word] = l_idx
                                local_word2idx[l_idx] = word
                                l_idx += 1
                            g2l[l_idx] = g_idx

        l = []
        adj_matrix = np.zeros((len(local_word2idx.keys()), len(local_word2idx.keys()))).astype('float32') ## adjacency matrix
        for i, item in enumerate(text):
            for j in range(len(text)):
                if i!=j:
                    y1 = (int(item['top'])+int(item['bottom']))//2 ## first coordinate
                    x1 = (int(item['left'])+int(item['right']))//2
                    y2 = (int(text[j]['top'])+int(text[j]['bottom']))//2 ## second coordinate
                    x2 = (int(text[j]['left'])+int(text[j]['right']))//2
                    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) ) ## distance between each centroid
                    i1 = local_word2idx[item['nlp']]
                    j1 = local_word2idx[text[j]['nlp']]
                    if dist<100: ## distance threshold
                        l.append(dist)
                        adj_matrix[i1][j1] = 1
        k+=1
    return global_word2idx



if __name__ == '__main__':
    embedding_model = 'small'
    global_word2idx = process1('processed_data_sample_100/train')
    embedding_matrix = build_embedding_matrix(global_word2idx, 300)
