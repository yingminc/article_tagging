from __future__ import division
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
import numpy as np
import operator
import sys
#sys.path.append('/var/sparkamplify/data_process/')
import db
from flashtext import KeywordProcessor
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
#import nltk
import string
from multiprocessing import cpu_count
import time
from math import sqrt, log
import os
import pickle
from datetime import datetime as dt

#from article_tagging_min import *

stemmer = WordNetLemmatizer().lemmatize
stop = stopwords.words('english') + list(string.punctuation)

kwrp=KeywordProcessor(case_sensitive=True)
kwsyn = {'ai':['artificial intelligence'],
         'ar':['augmented reality'],
         'smartglass':['google glass'],
         'biotech':['bioscience','biotech company'],
         'blockchain':['blockchains','blockchain technology'],
         'cryptocurrency':['cryptocurrencies','litecoin'],
         'ethereum':['eth'],
         'medium':['media'],
         'retail':['retailer'],
         'data science':['data scientist'],
         'drone':['drone delivery','delivery drone','drones'],'ecommerce':['e-commerce','e commerce']}
kwrp.add_keywords_from_dict(kwsyn)

# texr preprocess
def title_proc(title,kwrp=kwrp):
    return ','.join([stemmer(i) for i in word_tokenize(kwrp.replace_keywords(re.sub('[-/\']+','',title.lower())))
                     if (i not in stop)])

# sort dictionary by value
def dictsort(dicty,reverse=False,value=True):
    if value:
        return sorted(dicty.items(), key=operator.itemgetter(1),reverse=reverse)
    else:
        return [i[0] for i in sorted(dicty.items(), key=operator.itemgetter(1),reverse=reverse)]

# word frequency as dict
def freq_dict(corpus):
    CV=CountVectorizer(min_df=1,ngram_range=(1,2),token_pattern='(?u)[^(,\s)]\w*\s*\w*[^(,\s)]')

    voc_freq=CV.fit_transform(corpus).tocoo()
    vocab=CV.vocabulary_
    voc_reverse = {v:k for k,v in vocab.items()}

    total_count=np.unique(voc_freq.col,return_counts=True)
    for voc_ind,count in zip(total_count[0],total_count[1]):
        voc_count[voc_reverse[voc_ind]]=count
    return voc_count

# Tfidf train model
def text_keyword_train(train_text,vocabulary=None):
    td_model=TfidfVectorizer(stop_words=stop_words_list, min_df=3, vocabulary = vocabulary,
                       ngram_range=(1,2),sublinear_tf=True)
    tdm=td.fit_transform(train_text)
    ind_vocab={v:k for k,v in td.vocabulary_.items()}
    return td_model, tdm.tocoo(), ind_vocab

# Tfidf apply
def text_keyword_transform(target_text_list, td_model, fq_model):
    tdm_coo = td_model.transform(target_text_list).tocoo()
    fqm_coo = fq_model.transform(target_text_list).tocoo()
    row_len = len(target_text_list)
    ind_vocab={v:k for k,v in td_model.vocabulary_.items()}
    return tdm_coo, fqm_coo, row_len, ind_vocab

# text keyword
def text_keywords(tdm_coo, fqm_coo, row_len, ind_vocab, topn=5):
    kw_list=[]
    for row in range(row_len):
        if row not in tdm_coo.row:
            print('no text')
            kw_list.append('')
        else:
            fq_data=fqm_coo.data[np.where(fqm_coo.row==row)]
            fq_col=fqm_coo.col[np.where(fqm_coo.row==row)]
            kw_dict={}
            for i in zip(tdm_coo.col[np.where(tdm_coo.row==row)],tdm_coo.data[np.where(tdm_coo.row==row)]):
                fq=fq_data[np.where(fq_col==i[0])]
                if fq>0: #give higher weight to word appears in title
                    fq=fq[0]
                    fqw = fq+1
                    print(ind_vocab[i[0]],'freq: ',fqw)
                else:
                    fqw = 1
                    fq = 0
                if ind_vocab[i[0]].find(' ')>0: #bigram
                    kw_dict[ind_vocab[i[0]]]=((1+i[1]) * 1)*(fqw)
                    print(i[1])
                else:
                    kw_dict[ind_vocab[i[0]]]=((1+i[1]) * 2.5)*(fqw)
                    print(i[1])

            if len(kw_dict)<topn:
                print(dictsort(kw_dict, reverse=True, value=False))
                kws=','.join(dictsort(kw_dict, reverse=True, value=False))
            else:
                print(dictsort(kw_dict, reverse=True, value=False)[:topn])
                kws=','.join(dictsort(kw_dict, reverse=True, value=False)[:topn])
            kw_list.append(kws)
    return kw_list

def countmetrix(corpus,stop_words=stop_words_list, min_df=10,vocabulary=None):
    cectorizer=CountVectorizer(stop_words=stop_words, min_df=min_df,vocabulary=vocabulary,ngram_range=(1,2),
                               token_pattern='(?u)[^(,\s)]\w*\s*\w*[^(,\s)]')
    y_coo=cectorizer.fit_transform(corpus).tocoo()
    vocab=cectorizer.vocabulary_

    voc_reverse = {v:k for k,v in vocab.items()}
    voc_count = {}
    total_count=np.unique(y_coo.col,return_counts=True)
    for voc_ind,count in zip(total_count[0],total_count[1]):
        voc_count[voc_reverse[voc_ind]]=count

    return y_coo, vocab, voc_count

# co-occour prob keyword
def cooccour_prob(y_coo, vocab, voc_count, pos=None, neg=None, how_pos='union',
                topn=20, min_df=10, reverse=False, rank = False):
    # pos = postive keyword list
    # how_pos ={'union', 'intersection'}
    # neg = negative keyword list

    if (not pos):
        print('please give the postive kw')
    else:
        if pos: #add bigram process
            ind=0
            for cat in pos:
                cat = stemmer(cat)
                if cat in vocab:
                    if ind==0:
                        articles = y_coo.row[np.where(y_coo.col==vocab[cat])]
                        ind+=1
                    else:
                        if how_pos=='union':
                            articles = y_coo.row[np.where((y_coo.col==vocab[cat])|(np.isin(y_coo.row,articles)))]
                        elif how_pos=='intersection':
                            articles = y_coo.row[np.where((y_coo.col==vocab[cat])&(np.isin(y_coo.row,articles)))]
                        else:
                            print('pls choose how to select from positive ketwords /nhow_pos =\{\'union\', \'intersection\'\}')
                else:
                    print(cat+' is not in dict')
            if ind == 0:
                return 'no valid postive keywords'

        if neg:
            for ncat in neg:
                ncat = stemmer(ncat)
                neg_articles =y_coo.row[np.where(y_coo.col==vocab[ncat])]
                articles = articles[np.where(np.isin(articles,neg_articles,invert=True))]

        kw_collect = y_coo.col[np.where(np.isin(y_coo.row,articles))]

        kw_count=np.unique(kw_collect,return_counts=True)
        voc_reverse = {v:k for k,v in vocab.items()}
        kw_prob={}
        for voc_ind, count in zip(kw_count[0],kw_count[1]):
            if not voc_reverse[voc_ind].isdigit(): #BM25
                if (voc_count[voc_reverse[voc_ind]]>=min_df) and (count>2) \
                and (log(count/(voc_count[voc_reverse[voc_ind]])+1)*sqrt(count)>0.05) :
                    if reverse:
                        kw_prob[voc_reverse[voc_ind]]=(log(count/len(np.unique(articles))+1)*sqrt(count),count,
                                                   voc_count[voc_reverse[voc_ind]])

                    else:
                        kw_prob[voc_reverse[voc_ind]]=(log(count/(voc_count[voc_reverse[voc_ind]])+1)*sqrt(count),count,
                                                       voc_count[voc_reverse[voc_ind]])

        if rank:
            return dictsort(kw_prob,reverse=True)[:topn]
        else:
            return cooccour_prob
