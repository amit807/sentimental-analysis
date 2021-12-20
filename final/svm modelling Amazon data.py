#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import nltk
import string
pd.options.mode.chained_assignment = None
dataframe = pd.read_csv('book_review.csv').drop_duplicates().reset_index(drop=True)
df = dataframe[["review_text"]]
df["review_text"] = df["review_text"].astype(str)
dataframe.head()


# In[2]:


list(dataframe.columns)


# In[3]:


df.shape


# In[4]:


df["text_lower"] = df["review_text"].str.lower()
df.head()


# In[5]:


df.size


# In[6]:


df.shape


# In[7]:


list(df.columns)


# In[8]:


import spacy


# In[9]:


PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text_lower):
    """custom function to remove the punctuation"""
    return text_lower.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

df["text_wo_punct"] = df["text_lower"].apply(lambda text_lower: remove_punctuation(text_lower))
df.head()


# In[10]:


from nltk.corpus import stopwords


# In[11]:


stopwords


# In[12]:


nltk.download('stopwords')


# In[13]:


stopwords.words('English')


# In[14]:


", ".join(stopwords.words('english'))


# In[15]:


STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

df["text_wo_stop"] = df["text_wo_punct"].apply(lambda text: remove_stopwords(text))
df.head()


# In[16]:


from collections import Counter
cnt = Counter()
for text in df["text_wo_stop"].values:
    for word in text.split():
        cnt[word] += 1
        
cnt.most_common(10)


# In[17]:


FREQWORDS = set([w for (w, wc) in cnt.most_common(10)])
def remove_freqwords(text):
    """custom function to remove the frequent words"""
    return " ".join([word for word in str(text).split() if word not in FREQWORDS])

df["text_wo_stopfreq"] = df["text_wo_stop"].apply(lambda text: remove_freqwords(text))
df.head()


# In[18]:


n_rare_words = 10
RAREWORDS = set([w for (w, wc) in cnt.most_common()[:-n_rare_words-1:-1]])
def remove_rarewords(text_wo_stopfreq):
    """custom function to remove the rare words"""
    return " ".join([word for word in str(text_wo_stopfreq).split() if word not in RAREWORDS])

df["text_wo_stopfreqrare"] = df["text_wo_stopfreq"].apply(lambda text_wo_stopfreq: remove_rarewords(text_wo_stopfreq))
df.head()


# In[19]:


from nltk.stem.porter import PorterStemmer

# Drop the four columns 
df.drop(["text_wo_stopfreq", "text_lower", "text_wo_punct","text_wo_stop"], axis=1, inplace=True) 

stemmer = PorterStemmer()
def stem_words(text_wo_stopfreqrare):
    return " ".join([stemmer.stem(word) for word in text_wo_stopfreqrare.split()])

df["text_stemmed"] = df["text_wo_stopfreqrare"].apply(lambda text_wo_stopfreqrare: stem_words(text_wo_stopfreqrare))
df.head()


# In[20]:


nltk.download('wordnet')


# In[21]:


from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text_stemmed):
    return " ".join([lemmatizer.lemmatize(word) for word in text_stemmed.split()])

df["text_lemmatized"] = df["text_stemmed"].apply(lambda text_stemmed: lemmatize_words(text_stemmed))
df.head()


# In[22]:


df.head(100)


# In[23]:


# removings urls, there are no emojis and emoticons

def remove_urls(text_lemmatized):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text_lemmatized)

df["text_url_rmv"] = df["text_lemmatized"].apply(lambda text_lemmatized: remove_urls(text_lemmatized))
df.head()


# In[24]:


#removing tags

def remove_html(text_url_rmv):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text_url_rmv)

df["text_tags_rmv"] = df["text_url_rmv"].apply(lambda text_url_rmv: remove_html(text_url_rmv))
df.head()


# In[25]:


#spelling check
from spellchecker import SpellChecker

spell = SpellChecker()
def correct_spellings(text_tags_rmv):
    corrected_text = []
    misspelled_words = spell.unknown(text_tags_rmv)
    for word in str(text_tags_rmv).split():
        if word in misspelled_words:
            corrected_text.append(spell.correction(word))
        else:
            corrected_text.append(word)
    return " ".join(corrected_text)

df["Final_text"] = df["text_tags_rmv"].apply(lambda x: correct_spellings(x))


# In[26]:


list(df.columns)


# In[27]:


df.drop(['review_text','text_wo_stopfreqrare','text_stemmed','text_lemmatized','text_url_rmv','text_tags_rmv'], axis=1, inplace=True)
df.head()


# In[28]:


#calculating polarity

from textblob import TextBlob

def getSubjectivity(Final_text):
    return TextBlob(Final_text).sentiment.subjectivity
    
def getPolarity(Final_text):
    return TextBlob(Final_text).sentiment.polarity

df ['polarity'] = df['Final_text'].apply(getPolarity)
df['subjectivity'] = df['Final_text'].apply(getSubjectivity)

def getAnalysis(polarity):
    if polarity <= 0.01:
        return 'Negative'
    elif polarity >= 0.01:
        return 'Positive'
    else:
        return 'Neutral'
    
df['Analysis_labels'] = df['polarity'].apply(lambda x: getAnalysis(x))
        


# In[29]:


df.head(100)


# In[30]:


dataset=df[['Final_text','polarity','Analysis_labels']]
dataset.head()


# In[31]:


dataset.Analysis_labels.unique()


# # support vector machine

# In[32]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df0 =dataset[dataset.Analysis_labels=='Positive']
df1 =dataset[dataset.Analysis_labels=='Negative']
df2 =dataset[dataset.Analysis_labels=='Neutral']
df0.head()


# In[33]:


#plt.scatter(df0['Final_text'],df0['Analysis_labels'],colors='green')
#plt.scatter(df1['Analysis_labels'])


# In[34]:


from sklearn.feature_extraction.text import TfidfVectorizer
import time
from sklearn import svm
from sklearn.metrics import classification_report
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)


# In[35]:


from sklearn.model_selection import train_test_split
x = vectorizer.fit_transform(dataset['Final_text'])
y = vectorizer.transform(dataset['Analysis_labels'])


# In[36]:


x


# In[37]:


y=dataset['Analysis_labels']


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=5)


# In[39]:


x_train.shape


# In[40]:


from sklearn import svm
from sklearn.svm import SVC
model = SVC(C=70,kernel='linear',gamma='auto')


# In[41]:


model.fit(x_train, y_train)


# In[42]:


model.score(x_test, y_test)


# In[43]:


import pickle
pickle.dump(model, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))


# In[44]:


review = """SUPERB, I AM IN LOVE IN THIS book"""
review_vector = vectorizer.transform([review]) # vectorizing
print(model.predict(review_vector))


# In[45]:


import joblib
joblib.dump(vectorizer,'model_vectorizer.pkl')

object = pd.read_pickle(r'model.pkl')


# In[ ]:




