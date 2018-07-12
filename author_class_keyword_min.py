# coding: utf--8
from __future__ import division
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
import numpy as np
import operator
import sys
sys.path.append('/var/sparkamplify/data_process/')
import db
from flashtext import KeywordProcessor
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
#import nltk
import string
import time
from math import sqrt, log
import math
import os
import pickle

from article_tagging_min import *


PROJECT_PATH ='/var/sparkamplify/data_process/'

DATE='0613'
W2V = 'W2V/W2V1124_Skip_mc5_w5_77m.bin'
BIGRAM_FILE = 'category_reference/bigram/bigrams_2.dat'
STOP_WORD_FILE = 'category_reference/stop_word/stop_words.dat'
KEYWORD_REPLACER_FILE = 'category_reference/keyword_replacer.pickle'
KEYWORD_EXTRACTOR_FILE = 'category_reference/keyword_extract_model/keyword_extract_{}.pickle'.format(DATE)
KEYWORD_COUNTOR_FILE = 'category_reference/keyword_extract_model/keyword_freq_{}.pickle'.format(DATE)
VOC_IND_DICT = 'category_reference/keyword_extract_model/keyword_ind_dict_{}.pickle'.format(DATE)
CLASS_KW_DICT ='category_reference/category_kw/dict_relate_words_{}_2018.pickle'.format(DATE)

s3 = boto3.resource('s3', aws_access_key_id='AKIAJH6SLU7BCLAS2DBA',
                        aws_secret_access_key='oQ0K5D8HTPBzHx9Le+IgcfvSryjXnr8dLQrBOyE+',
                        region_name='ap-northeast-1')

kwrp = pickle.load(open(os.path.join(PROJECT_PATH,KEYWORD_REPLACER_FILE),'rb'))
stemmer = WordNetLemmatizer().lemmatize
stop = stopwords.words('english') + list(string.punctuation)+['text','nan']

# data extract from s3========================================================================

def to_article_df(df): # use both 'title'&'text' for tagging
    if set(['keyword','name','title']).issubset(df.columns) and len(df.columns)<=4:
        a=df[['name','keyword','title']]
        a['id']=a.index
        return a
    else:
        #Harry add For CES
        article=author_article_item_extract_CES(df)
        if('title' not in article.columns):
            df['title']=np.nan
        a=article[['author_source','content','title']]
        a=a.rename(columns = {'author_source':'name','content':'text','title':'title'})
        a['id']=a.index
        return a


def author_article_item_extract(df):
    article=df[df.author_source.notnull() & df.route.notnull()]
    author=df[df.name.notnull() & ~df.route.notnull()]
    return author,article

def author_article_item_extract_CES(df):

    article = df[df.author_source.notnull()]

    # Harry add processing author_name is none
    #if 'author_source' not in df.columns:
    #    df['author_source']=np.nan

    #da.loc[da.author_source.isnull(),'author_source']=''

    return article


def get_date_file(bucketname,date,spidername):
    list=[]
    for i in s3.Bucket(bucketname).objects.all():

        if i.key[i.key.find('/')+1:i.key.find('T')]<date and '%s/'%spidername in i.key:
            list.append(i.key)
    return list

def read_s3_file(bucketname,filename):
    return  io.StringIO(s3.Object(bucketname,filename).get()['Body'].read().decode('utf-8'))

#==== data type 2017-11-20======

def read_article(date,bucketname='scraping-new-proccess',spidername='gt'):
    df=0
    for file_name in get_date_file(bucketname,date,spidername):
        if file_name.endswith('.csv'):
            try:
                if type(df)==int:
                    print('start')
                    df=pd.read_csv(read_s3_file(bucketname,file_name))
                else:
                    df=df.append(pd.read_csv(read_s3_file(bucketname,file_name)),ignore_index=True)
            except:
                print('%s--error-----'%file_name)
    return df


def article_lemmatizer_proccess(_article):
    try:
        filter_regex = re.compile(r'[^a-zA-Z0-9 -_  ]')
        _article = filter_regex.sub(' ', _article).lower()

        _article = _article.split(' ')

        #Start lemma.
        lemma_article = ""
        for word in _article:
            #(1)remove punctuation
            word = filter_regex.sub(' ', word).lower()
            #(2)remove ' ' '\t' '\n'
            word = word.replace('\t','').replace('\n','').replace(' ','').replace('-',' ').replace('_',' ')
            #(3)lemma + to lower
            word = (stemmer(word)).lower().strip(' ')
            #combine all word
            if word:
                lemma_article = lemma_article +' '+stemmer(word)
        return kwrp.replace_keywords(lemma_article)
    except:
        print(_article)

        return ''

def text_proc(text):
    return ' '.join([stemmer(i) for i in
    word_tokenize(kwrp.replace_keywords(re.sub('[-_]',' ',re.sub('[/\'\t\n]+','',text.lower()))))])

def title_proc(title):
    return ','.join([stemmer(i) for i in word_tokenize(kwrp.replace_keywords(re.sub('[-/\']+','',title.lower())))
                     if (i not in stop)])

def stopword_building():
    print('making stopword_list')
    stop_words_list = []
    #Stop word building
    with open(os.path.join(PROJECT_PATH,STOP_WORD_FILE),encoding='utf-8') as file:
        for stop_word in file:
            stop_word = stop_word.strip("\n").strip('  ').strip(' ')
            stop_words_list.append(stop_word)

    return stop_words_list

def bigram_proc():
    print('making bigram_list')
    ngrams_dict = {}
    with open(os.path.join(PROJECT_PATH,BIGRAM_FILE),'r',encoding='utf-8') as file:

        for line in file:
            data = line.strip('\n').strip(' ').split('\t')
            word = data[0]
            COUNT= data[1]
            ngrams_dict[word] = COUNT

    bigram_list = ngrams_dict.keys()
    file.close()
    return bigram_list

#========================little func for tagging=====================
# order keys/items in dict
def dictsort(dicty,reverse=False,value=True):
    if value:
        return sorted(dicty.items(), key=operator.itemgetter(1),reverse=reverse)
    else:
        return [i[0] for i in sorted(dicty.items(), key=operator.itemgetter(1),reverse=reverse)]

def word_reform(word):
    return kwrp.replace_keywords(stemmer(word))


stop_words_list = stopword_building()

#make relative words metrix
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

def freq_dict(corpus):
    CV=CountVectorizer(min_df=1,ngram_range=(1,2),token_pattern='(?u)[^(,\s)]\w*\s*\w*[^(,\s)]')

    voc_freq=CV.fit_transform(corpus).tocoo()
    vocab=CV.vocabulary_
    voc_reverse = {v:k for k,v in vocab.items()}
    voc_count={}
    total_count=np.unique(voc_freq.col,return_counts=True)
    for voc_ind,count in zip(total_count[0],total_count[1]):
        voc_count[voc_reverse[voc_ind]]=count
    return voc_count

#find keyword for each category
def cat_keyword(y_coo, vocab, voc_count, pos=None, neg=None, how_pos='union',
                topn=20, min_df=10, reverse=False, rank = False):
    # pos = postive keyword list
    # how_pos ={'union', 'intersection'}
    # neg = negative keyword list

    if (not pos):
        print('please give postive kw')
    else:
        if pos: #add bigram process
            ind=0
            for cat in pos:
                #cat = stemmer(cat)
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
                #ncat = stemmer(ncat)
                neg_articles =y_coo.row[np.where(y_coo.col==vocab[ncat])]
                articles = articles[np.where(np.isin(articles,neg_articles,invert=True))]

        kw_collect = y_coo.col[np.where(np.isin(y_coo.row,articles))]

        kw_count=np.unique(kw_collect,return_counts=True)
        voc_reverse = {v:k for k,v in vocab.items()}
        kw_prob={}
        for voc_ind, count in zip(kw_count[0],kw_count[1]):
            if not voc_reverse[voc_ind].isdigit():
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
            return kw_prob

#======================MAIN FUNC==================================

def author_keyword_category(df,dp): #need 'name', 'keyword', 'title' columns
    start=time.time()

    df.loc[df.title.notnull(),'title']=df.loc[df.title.notnull(),'title'].apply(lambda x: title_proc(x))
    print('process title:', time.time() - start, 'sec')

    #combine title and keywords
    df['kw_set']=(df.title.fillna('')+','+df.keyword.fillna('')).apply(lambda x: ','.join(set(x.split(','))))

    #combine articles of same author
    author_kw={}
    author_n={}
    for author in df.loc[df.name.notnull(),'name'].unique():
        author_n[author] = len(df.loc[df.name==author,'kw_set'])
        author_kw[author]=','.join(list(df.loc[df.name==author,'kw_set']))

    del df

    td_model=pickle.load(open(os.path.join(PROJECT_PATH,KEYWORD_EXTRACTOR_FILE),'rb'))
    voc_dict=pickle.load(open(os.path.join(PROJECT_PATH,VOC_IND_DICT),'rb'))

    #the dict for category related words
    sim_kw = pickle.load(open(os.path.join(PROJECT_PATH,CLASS_KW_DICT),'rb'))

    #start tagging and get author kw
    author_df = {}
    for author, kws in author_kw.items():
        start=time.time()
        #precess the article number factor
        if log(author_n[author])>1.5:
            article_n_log = log(author_n[author])
        else:
            article_n_log =1.5
        if  kws.replace(',','') == '':
            pass
        else:
            print('author: ',author)
            #get author keyword
            len_row, coo= text_keyword_transform([kws],td_model)
            voc_freq=freq_dict([kws])

            kws_wight={voc_dict[k]: (math.pow(voc_freq[voc_dict[k]],1/2)) * (1+v) \
                       for k ,v in zip(coo.col,coo.data) if voc_dict[k] in voc_freq.keys()}

            keywords={k:v for k, v in dictsort(kws_wight,reverse=True,value=True)[:10]}
            print('keywords: ',keywords)

            kws=kws.split(',')

            #get category score
            cat_score = {}
            for cat in sim_kw.keys():
                if cat == 'other':
                    pass
                else:
                    Cat =cat.capitalize()
                    cat_score[Cat]=0
                    for kw in set(kws):
                        kw =word_reform(kw)
                        if kw in sim_kw[cat].keys():
                            cat_score[Cat]+=sim_kw[cat][kw][0]

            cat_score['Other']=0

            # assign 'other' category
            if (max(cat_score.values())/article_n_log)<3:
                if article_n_log==1.5:
                    cat_score['Other']=3*1.5
                else:
                    cat_score['Other']=author_n[author]

            #set treshold and adjust by article number
            true_cats={}
            for Cat in cat_score.keys():
                if (cat_score[Cat]/article_n_log)>=3:
                    true_cats[Cat]=int((cat_score[Cat]/article_n_log)*7)-1
            del cat_score

            if len(true_cats)>5:
                for fakecat in dictsort(true_cats,reverse=True,value=False)[5:]:
                    del true_cats[fakecat]

            true_cats = str(true_cats).replace('\'','"')

            print('categories: ',true_cats)

            print('process time: ', time.time()-start, ' sec','\n')

            author_df[author]={'keyword':str(keywords).replace('\'','"'), 'category':true_cats}

    d_processed = pd.DataFrame.from_dict(author_df).T.reset_index().rename(columns={'index':'name'})
    d = pd.merge(d_processed, dp, on='name', how='right')

    return d
