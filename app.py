from flask import Flask, request,render_template,redirect, url_for
import requests,re
from bs4 import BeautifulSoup as bs
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # for regex
# NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
# BOW
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score
# pkl
import pickle

# headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36'}

word_list = pickle.load(open('bow1.pkl','rb'))
clf = pickle.load(open('model1.pkl','rb'))

def scraped(url):
    req = requests.get(url).text
    idx,title,year = [],[],[]
    pattern = r'[0-9]{4}'
    soup = bs(req,'html.parser')
    table = soup.find('table').find_all('tr')
    if len(table) < 10:
        for i in range(len(table)):
            idx.append(str(table[i].find_all('a')[1]['href']).split('/')[-2])
            title.append(table[i].find_all('a')[1].text.strip())
            year.append(re.findall(pattern,table[i].find_all('td')[1].text)[0])
    else:
        for i in range(len(table[:11])):
            idx.append(str(table[i].find_all('a')[1]['href']).split('/')[-2])
            title.append(table[i].find_all('a')[1].text.strip())
            year.append(re.findall(pattern,table[i].find_all('td')[1].text)[0])
    return idx,title,year

def scraped_revs(url1,url2):
    req = requests.get(url1).text
    req2 = requests.get(url2).text

    genres,revs = [],[]
    soup = bs(req,'html.parser')
    soup2 = bs(req2,'html.parser')        # For reviews
    title = soup.find('div', {"class":"title_wrapper"}).find_all('h1')[0].text
    try:
        ratings = soup.find('div', {"class":"ratingValue"}).find_all('span')[0].text
    except:
        ratings = "Not Rated"
    try:
        duration = soup.find('div', {"class":"subtext"}).find_all('time')[0].text.strip()
    except:
        duration = "No Information Available"
    try:
        lst = soup.find('div', {"class":"subtext"}).find_all('a')
        for i in range(len(lst)-1):
            genres.append(lst[i].text)
        gen = ",".join(genres)
    except:
        gen = "No Information Available"
    try:
        date = lst[-1].text.strip()
    except:
        date = "No Information Available"
    try:
        image = soup.find('div', {"class":"poster"}).find_all('img')[0]['src'] 
    except:
        image = "No Image Available"
    try:
        rev_div = soup2.find('div',{"class":"lister-list"}).find_all('div',{"class":"lister-item-content"})
        for i in range(len(rev_div)):
            rev_rate = rev_div[i].find_all('span')[1].text
            if len(rev_rate.split(" ")) == 1:
                rev_ratings = rev_rate
            else:
                rev_ratings = "Not rated"
            rev_title = rev_div[i].find_all('a',{"class":"title"})[0].text.strip()
            user = rev_div[i].find_all('a')[1].text.strip()
            rev_rev = rev_div[i].find('div',{"class":"text show-more__control"}).text.strip()

            rev_d = {'ratings':rev_ratings,'title':rev_title,'user':user,'review':rev_rev}
            revs.append(rev_d)
    except:
        revs.append("No Reviewers yet")
    return title,ratings,duration,gen,date,image,revs

# TEXT PREPROCESSING FUNCTIONS
def clean(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned,'',text)

def is_special(text):
    rem = ''
    for i in text:
        if i.isalnum():
            rem = rem + i
        else:
            rem = rem + ' '
    return rem

def to_lower(text):
    return text.lower()

def rem_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stop_words]

def stem_txt(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/choices', methods=['POST'])
def choices():
    title = request.form.get('movie')
    if len(title.split()) == 1:
        url = 'https://www.imdb.com/find?s=tt&q={}&ref_=nv_sr_sm'.format(title)
    else:
        title_str = "+".join(title.split())
        url = 'https://www.imdb.com/find?s=tt&q={}&ref_=nv_sr_sm'.format(title_str)
    idx,movie,yor = scraped(url)
    return render_template('choices.html',idx=idx,movie=movie,yor=yor)

@app.route('/review/<string:type>')
def review(type):
    url1 = "https://www.imdb.com/title/{}/?ref_=fn_tt_tt_1".format(type)
    url2 = "https://www.imdb.com/title/{}/reviews?ref_=tt_urv".format(type)
    title,ratings,duration,genres,date,image,revs = scraped_revs(url1=url1,url2=url2)
    labels = []

    for r in revs:
        rv_txt = r['review']
        f1 = clean(rv_txt)
        f2 = is_special(f1)
        f3 = to_lower(f2)
        f4 = rem_stopwords(f3)
        f5 = stem_txt(f4)

        inp = []
        for i in word_list:
            inp.append(f5.count(i[0]))
        label = clf.predict(np.array(inp).reshape(1,1000))[0]

        rate = r['ratings']
        if rate in ['8','9','10']:
            labels.append(1)
        elif rate in ['1','2','3']:
            labels.append(0)
        else:
            if label == 1:
                labels.append(label)
            else:
                labels.append(-1)
    if len(labels):
        sent_score = (len([i for i, x in enumerate(labels) if x == 1])/len(labels))*10
    else:
        sent_score = 0
    return render_template('review.html',title=title,ratings=ratings,duration=duration,genres=genres,date=date,image=image,revs=revs,labels=labels,sent_score=sent_score)

if __name__=="__main__":
    app.run(debug=True)