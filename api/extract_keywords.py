# This code extracts keywords from the articles and inputs them into the dynamodb based on a dynamodb trigger
# updates the keyword counter table and the articles table
# import relevant packages
from datetime import timedelta, date
from fuzzywuzzy import fuzz
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import pandas as pd
import numpy as np
import datetime
import glob
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))


# stopwords
stop_words = stopwords
extra_stopwords = ['shares', 'person', 'useful', 'govtech', 'cio', 'yonhap', 'size', 'tackle', 'right', 'day', 'tried', 'tested', 'make', 'sure', 'used', 'help', 'yesterday', 'today', 'tomorrow', 'percent', 'per', 'cent', 'could', 'many', 'add', 'use', 'need', 'goods', 'million', 'thousand', 'company', 'retailers', 'saw', 'see', 'new', 'like', 'today', 'tomorrow', 'guide',
                   'people', 'want', 'yet', 'way', 'time', 'back', 'whether', 'if', 'yes', 'older', 'noted', 'went', 'told', 'tell', 'younger', 'another', 'worth', 'noting', 'well', 'called', 'named', 'never', 'lee', 'quah', 'ong', 'ng', 'lim', 'tan', 'shared', 'says', 'say', 'said', 'cio', 'cios', 'month', 'top', 'world', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'january', 'february', 'march', 'april', 'may', 'june', 'july',
                   'august', 'september', 'october', 'november', 'december', 'month', 'months', 'years', 'year', 'near', 'also', 'would', 'able']
for word in extra_stopwords:
    stop_words.add(word)


def add_keywords(tags,  keyword_dict):
    '''
    input: tags in either a list or string format, empty or filled keyword dict, counts the number of words 
    outputs: a dictionary of keyword counts
    '''
    if type(tags) != list:
        tag_split = tags.split(', ')
        for word in tag_split:
            keyword_dict[word.strip().lower()] = keyword_dict.get(
                word.strip().lower(), 0) + 1
    return keyword_dict


def remove_similar_keywords(keyword_list):
    """takes in keyword_list and removes similar words keeping only one word out of the pair(eg, vaccines, vaccine)"""

    seen = []
#     keyword_list = keyword_list.split(', ')
    for i in keyword_list:
        for k in keyword_list:
            if (fuzz.ratio(i, k) >= 82) & (i != k):
                seen.append(k)
#                 print(i,k)
#                 print(fuzz.ratio(i ,k))
    seen = [seen[i] for i in range(len(seen)) if i % 2 != 0]
    res = [e for e in keyword_list if e not in seen]
    return res

# another method of cleaning text but this removes too many impt words


def cleaning(text):
    text = ''.join(text)
    text = text.strip()
    text = " ".join(text.split())
    text = re.sub("[^0-9a-zA-Z-]", " ", text)
    text = re.sub('[0-9]{2,4}', ' ', text)
    article_text = re.sub(r'\s+', ' ', text)
#     article_text = ' '.join([w.lower() for w in article_text.split() if w not in stopwords])
    tokens = [w.lower() for w in article_text.split() if w not in stop_words]
#     doc =nlp(article_text)
#     doc = [t.text if not t.ent_type_ else t.ent_type_ for t in doc]
#     tokens = [word for word in doc if word not in ['DATE','PERCENT', 'TIME']]
    tags = nltk.pos_tag(tokens)
    nouns = " ".join([word.lower() for word, pos in tags if pos not in [
                     'MD', 'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'RB', 'RBR', 'RBS', 'IN'] and len(word) > 1])
#     article_text = unicodedata.normalize("NFKD", article_text)
    return nouns


def transform_single_text_cv(new_data_text):
    '''
    Takes in the text columns from both new data and train data and transforms data to outputs tf-idf of new data text

    new_data_text:  text column of new data
    train_data_text: text column of train set data
    '''
    new_data_text2 = cleaning(new_data_text)
#     new_data_text2 = thorough_cleaning(new_data_text)
#     new_data_text = new_data_text.apply(lambda x: thorough_cleaning(x))
    cv_vect = CountVectorizer(
        analyzer='word', token_pattern=r'\w{1,}', max_features=5000, stop_words=stop_words)
#     cv_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000,stop_words=stop_words)
    cv_vect.fit([new_data_text2])
    cv = cv_vect.transform([new_data_text2])

    return cv, cv_vect


def transform_single_text_ngram_cv(new_data_text, ngram_range=(1, 2)):
    '''
    Takes in the text columns from both new data and train data and transforms data to outputs tf-idf of new data text

    new_data_text:  text column of new data
    train_data_text: text column of train set data
    '''
    new_data_text2 = cleaning(new_data_text)
#     new_data_text2 = thorough_cleaning(new_data_text)
#     cv_vect_ngram = TfidfVectorizer(analyzer='word',  token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=5000,stop_words=stop_words)
    cv_vect_ngram = CountVectorizer(
        analyzer='word',  token_pattern=r'\w{1,}', ngram_range=ngram_range, max_features=5000, stop_words=stop_words)
    cv_vect_ngram.fit([new_data_text2])
    cv = cv_vect_ngram.transform([new_data_text2])
    return cv, cv_vect_ngram


def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)


def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""

    # use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []

    for idx, score in sorted_items:
        fname = feature_names[idx]

        # keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    # create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results = {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]] = score_vals[idx]

    return results


def get_keywords_from_single_text(new_data_text, method='cv', ngram_range=(1, 2), topn=10):

    if method == 'cv':
        # transform text to tf-idf
        cv, cv_vect = transform_single_text_cv(new_data_text)
#     print(vect)
    elif method == 'ngram':
        cv, cv_vect = transform_single_text_ngram_cv(
            new_data_text, ngram_range=ngram_range)

    # create dictionary to translate id to name
    id2name = {idx: name for idx, name in enumerate(
        cv_vect.get_feature_names())}
    # sort the tf-idf vectors by descending order of scores
    sorted_items = sort_coo(cv.tocoo())

    # extract only the top n; n here is 10
    keywords = extract_topn_from_vector(id2name, sorted_items, topn)
    return keywords

# to remove duplicate/common ngrams


def clean_up(ngrams):
    '''Takes in ngram data remove duplicate unigram and bigram eg. 'digital','government' vs 'digital government'''
    seen = set()
    for ngram in ngrams:
        if ' ' in ngram:
            seen = seen.union(set(ngram.split()))
    return [ngram for ngram in ngrams if ngram not in seen]


def extract_single_text_ngram(data, ext_keywords, topn=10):
    """
    takes in a single document and extracts the keywords based on count vectorizer 
    Allows user to input list of external keywords to include in the final keyword set 
    returns a list of words
    """
    # extracting bigrams from article (10 keywords default)
    # limit bigrams as they repeat common words
    keywords_tuple = set(list(get_keywords_from_single_text(
        data, 'ngram', ngram_range=(2, 2)))[:5])
    extended_tuple_words = [
        word for word in keywords_tuple if word in ext_keywords]
    # extract unigrams (basically ti-idf)
    keywords = list(get_keywords_from_single_text(
        data, 'ngram', ngram_range=(1, 1), topn=30))
    extended_words = [word for word in keywords if word in ext_keywords]

    # adds this extra keywords to the set
    keywords = set(keywords[:topn])
    keywords = keywords.union(keywords_tuple)
    # replace the space in between words as to standardize keywords with categories
    keywords = set([word for word, pos in pos_tag(keywords) if pos not in [
                   'MD', 'VB', 'VBP', 'VBD', 'VBG', 'VBN', 'VBZ', 'RB', 'RBR', 'RBS', 'IN']])
#     print(keywords)
    # in case some important words are removed
    keywords = keywords.union(set(extended_words))
    keywords = keywords.union(set(extended_tuple_words))
    words = remove_similar_keywords(keywords)
#     words = clean_up(keywords)
    return ', '.join(list(words))
