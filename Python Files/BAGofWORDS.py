#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import re
import nltk
from sklearn.datasets import load_files
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


data=pd.read_csv('train.csv')


# In[3]:


X=data.text
y=data.label


# In[4]:



'''Lemmatization usually refers to doing things properly with the use of a vocabulary and
morphological analysis of words, normally aiming to remove inflectional endings only and 
to return the base or dictionary form of a word, which is known as the lemma .'''
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()


# In[5]:


''' cleaning, removing stop words'''
#for text
document_X = []
for sen in range(0, len(X)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    document_X.append(document)


# In[6]:


#for label
document_Y= []
for sen in range(0, len(y)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(y[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    document_Y.append(document)


# In[7]:


#splitting the data based on 70-30 ratio
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(document_X, document_Y, test_size=0.2, random_state=0)


# In[8]:


# Initialize the `count_vectorizer` 
count_vectorizer = CountVectorizer(stop_words='english')

# Fit and transform the training data 
count_train = count_vectorizer.fit_transform(X_train)  # Learn the vocabulary dictionary and return term-document matrix.
#to create sparse matrix--> count_train = count_vectorizer.fit_transform(X_train).todense()
# Transform the test set 
count_test = count_vectorizer.transform(X_test)

# Get the feature names of `count_vectorizer` 
print(count_vectorizer.get_feature_names())


# In[9]:


#tf_idf
'''In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical
statistic that is intended to reflect how important a word is to a document in a collection or corpus.'''

'''Term Frequency (tf): gives us the frequency of the word in each document
Inverse Data Frequency (idf): used to calculate the weight of rare words across all documents'''

'''Combining these two we come up with the TF-IDF score (w) for a word in a document'''

'''The output obtained is in the form of a skewed matrix, which is normalised to get the following result.'''

tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words='english')
# This removes words which appear in more than 70% of the articles

# Fit and transform the training data 
tfidf_train = tfidfconverter.fit_transform(X_train)

# Transform the testing data 
tfidf_test = tfidfconverter.transform(X_test)

# Get the feature names of `tfidf_vectorizer` 
print(tfidfconverter.get_feature_names())


# In[13]:


count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())
print(count_vectorizer.vocabulary_)
#CREATED BagOfWords


# In[11]:


tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidfconverter.get_feature_names())
print(tfidfconverter.vocabulary_)
#CREATED BagOfWords


# In[12]:


import pickle
#save BoW as a pickle object in Python
with open('bow', 'wb') as picklefile:
    pickle.dump(count_vectorizer.get_feature_names(),picklefile)

