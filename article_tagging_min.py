# coding: utf--8
import boto3
import pandas as pd
import numpy as np
import db
import codecs
import scipy
import io
import os
import nltk
import pickle
import flashtext
import re
import time
import string
import operator
#import json
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer


PROJECT_PATH ='/var/sparkamplify/data_process/'

#configure all the pretrain model

DATE = '0613'
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

def to_article_df(df,_type='other'): # use both 'title'&'text' for tagging
    if set(['text','name','title']).issubset(df.columns) and len(df.columns)<=4:
        a=df[['name','text','title']]
        a['id']=a.index
        return a
    else:
        #Harry add For CES
        if('author_source' not in df.columns):
            df['author_source']=np.nan
        article=author_article_item_extract_CES(df,_type)
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

def author_article_item_extract_CES(df,_type):

    if(_type=='placement'):
        article = df
    else:
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
        return kwrp.replace_keywords(lemma_article[1:])
    except:
        print(_article)

        return ''

def text_proc(text):
    try:
        return ' '.join([stemmer(i) for i in
        word_tokenize(kwrp.replace_keywords(re.sub('[-_]',' ',re.sub('[/\'\t\n]+','',text.lower()))))])
    except AttributeError:
        return ''

def title_proc(title, return_str=True):
    try:
        if return_str:
            return ','.join([stemmer(i) for i in word_tokenize(kwrp.replace_keywords(re.sub('[-/\']+','',title.lower())))
                             if (i not in stop)])
        else:
            return [stemmer(i) for i in word_tokenize(kwrp.replace_keywords(re.sub('[-/\']+','',title.lower()))) if (i not in stop)]
    except AttributeError:
        return ''

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

# train the tfidf model
def text_keyword_train(train_text_list,vocabulary=None):
    td=TfidfVectorizer(stop_words=stop_words_list, min_df=3, vocabulary = vocabulary, ngram_range=(1,2),sublinear_tf=True)
    tdm=td.fit_transform(train_text_list)
    ind_vocab={v:k for k,v in td.vocabulary_.items()}
    return td, tdm.tocoo(), ind_vocab

# fit the target data with model
def text_keyword_transform(target_text_list, td_model):
    return len(target_text_list), td_model.transform(target_text_list).tocoo()

# get keywords after fitting
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
                    #print(ind_vocab[i[0]],'freq: ',fqw)
                else:
                    fqw = 1
                    fq = 0
                if ind_vocab[i[0]].find(' ')>0: #bigram
                    kw_dict[ind_vocab[i[0]]]=((1+i[1]) * 1)*(fqw)
                    #print(i[1])
                else:
                    kw_dict[ind_vocab[i[0]]]=((1+i[1]) * 2.5)*(fqw)
                    #print(i[1])

            if len(kw_dict)<topn:
                #print(dictsort(kw_dict, reverse=True, value=False))
                kws=','.join(dictsort(kw_dict, reverse=True, value=False))
            else:
                #print(dictsort(kw_dict, reverse=True, value=False)[:topn])
                kws=','.join(dictsort(kw_dict, reverse=True, value=False)[:topn])
            kw_list.append(kws)
    return kw_list

def article_class(kws, other_threshold=1):
    sim_kw= pickle.load(open(os.path.join(PROJECT_PATH,CLASS_KW_DICT),'rb'))
    if kws == '':
        pass
    else:
        kws=kws.split(',')
        cat_score = {}
        for cat in sim_kw.keys():
            Cat =cat.capitalize()
            if cat == 'other':
                pass
            else:
                cat_score[Cat]=0
                for kw in set(kws):
                    kw =word_reform(kw)
                    if kw in sim_kw[cat].keys():
                        cat_score[Cat]+=sim_kw[cat][kw][0]
        if sum(cat_score.values())<other_threshold:
            cat_score['Other']=other_threshold
        else:
            cat_score['Other']=0
        return str(cat_score).replace('\'','"')



#=======================================MAIN FUNC===========================================

def Article_tagging(df, filename,other_threshold=1, topn=5): #use a pretrain model
    # w2v= Word2Vec.load(os.path.join(PROJECT_PATH,W2V))
    # wv=w2v.wv
    # stop_words_list=stopword_building()
    # bigram_list =bigram_proc()
    # voc_list = list(set([i for i in wv.vocab.keys()]+list(bigram_list)))
    td_model=pickle.load(open(os.path.join(PROJECT_PATH,KEYWORD_EXTRACTOR_FILE),'rb'))
    fq_model=pickle.load(open(os.path.join(PROJECT_PATH,KEYWORD_COUNTOR_FILE),'rb'))
    voc_dict=pickle.load(open(os.path.join(PROJECT_PATH,VOC_IND_DICT),'rb'))

    #df=to_article_df(df)

    print('Processing text')
    start=time.time()
    df['text']=df['text'].apply(article_lemmatizer_proccess)
    titlelist = df['title'].fillna('').tolist()
    Text_list=(df['title'].apply(article_lemmatizer_proccess)+' '+df.text).fillna('').tolist()
    pos_titlelist=[]
    error=0
    for title in titlelist:
        try:
            pos_title= (' ').join([i[0] for i in nltk.pos_tag(title_proc(title, return_str=False)) if i[1] in ['NN','NNP','VBG','NNS','NNPS']])
        except IndexError:
            error+=1
            pos_title=title
            print(error)
        pos_titlelist.append(pos_title)
    print('Done in {} sec'.format(time.time() - start))

    # transform input text
    print('Get keywords')
    start=time.time()
    row_len, tdm=text_keyword_transform(Text_list, td_model)
    fqm = td_model.transform(pos_titlelist).tocoo()
    df['keyword']=text_keywords(tdm, fqm, row_len, voc_dict, topn=topn)

    # get article class score
    df.loc[df.title.notnull(),'title']=df.loc[df.title.notnull(),'title'].apply(lambda x: title_proc(x))
    df['category']=(df.title.fillna('')+','+df.keyword.fillna('')).apply(lambda x: ','.join(set(x.split(','))))
    df['category']=df['category'].apply(lambda x: article_class(x,other_threshold=other_threshold))

    print('Done in {} sec'.format(time.time() - start))

    return df
