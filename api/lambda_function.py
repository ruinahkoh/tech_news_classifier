# -*- coding: utf-8 -*-
from datetime import datetime, date
from newspaper import Article
from config import API_KEY, DYNAMODB_ARTICLES_TABLE,BUCKET_NAME
from newsapi  import NewsApiClient
import boto3
import zipfile

s3 = boto3.resource('s3')
KEY = 'keyword_extraction/nltk_data.zip'
local_file_name = '/tmp/nltk_data.zip'
s3.Bucket(BUCKET_NAME).download_file(KEY, local_file_name)
# unzip nltk_data.zip in lambda store to tmp folder
with zipfile.ZipFile(local_file_name, 'r') as zip_ref:
    zip_ref.extractall('/tmp/')
# append nltk data path to lambda
import nltk
nltk.data.path.append("/tmp/nltk_data")

def get_key_word(url):
        key_words = []
        article = Article(url)
        article.download()
        # article.html
        article.parse()
        article.nlp()

        return article.text, article.summary
def get_news_articles(topic):
    # Init
    newsapi = NewsApiClient(api_key=API_KEY)
    
    # get everything using topic for query key
    articles= newsapi.get_everything(q=topic,language='en')
    print(articles)
    art=[]
    for i in range(len(articles['articles'])):
        title = articles['articles'][i]['title']
        url = articles['articles'][i]['url']
        blurp = articles['articles'][i]['description']
        imgurl = articles['articles'][i]['urlToImage']
        article_date = articles['articles'][i]['publishedAt']
        article_date = article_date.split('T')[0]
        source = articles['articles'][i]['source']['name'].lower()



        try:
            text, blurp1 = get_key_word(url)
        except IndexError:
            pass

        try:
            article_date = datetime.strptime(article_date, "%Y-%m-%d")
            article_date = datetime.strftime(article_date, "%d/%m/%Y")
        except IndexError:
            article_date = date.today().replace(day=1)
            article_date = datetime.strftime(article_date, "%d/%m/%Y")
        art.append(
        {
            'title': title,
            'imgurl': imgurl,
            'date': article_date,
            'blurp': blurp,
            'url': url,
            'text': text,
            'category': topic,
            'source': source,
            'tags': None
        })
    return art

    
dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table(DYNAMODB_ARTICLES_TABLE)
def lambda_handler(event, context):
    topic=event['topic']
    items = get_news_articles(topic)
    for item in items:
        table.put_item(Item=item)

