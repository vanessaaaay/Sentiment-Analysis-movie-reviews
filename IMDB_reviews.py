#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train = pd.read_csv("./labeledTrainData.tsv",header=0,delimiter='\t')
test = pd.read_csv("./testData.tsv",header=0,delimiter='\t')
unlabeled = pd.read_csv("./unlabeledTrainData.tsv",header=0,delimiter='\t',quoting=3)
train["id"][0]


# In[3]:


unlabeled.shape


# In[8]:


train.columns.values


# In[15]:


train["review"][0]


# In[13]:


from bs4 import BeautifulSoup


# In[ ]:





# In[14]:


import nltk

from nltk.corpus import stopwords
stopwords.words('english')[:20]


# In[19]:


import re
def review_to_text (raw_text):
    review_text = BeautifulSoup(raw_text,"html").get_text()
    letters_only = re.sub('[^a-zA-Z]'," ",review_text)
    into_words = letters_only.lower().split()
    stop_words = set(stopwords.words('english'))
    meaningful_text = []
    for w in into_words:
        if w not in stop_words:
            meaningful_text.append(w)
    return " ".join(meaningful_text)


# In[19]:



X_test = []
for review in test["review"]:
    X_test.append(review_to_text(review))


# In[20]:


X_train = []
for review in train["review"]:
    X_train.append(review_to_text(review))
y_train = train["sentiment"]


# In[23]:





# In[37]:


# *******************先用tfidf的方法*************************

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

mnb_tfidf = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('mnb', MultinomialNB())
])

grid_params = {
  'mnb__alpha': [0.1,1.0,10.0],
  #'tfidf__max_df': [0.1, 1, 10],
  'tfidf__binary': [True, False],
  'tfidf__ngram_range': [(1,1),(1,2)]
}
clf = GridSearchCV(mnb_tfidf, grid_params, cv=4, n_jobs=-1, verbose=1 )
clf.fit(X_train, y_train)

print("Best Score: ", clf.best_score_)
print("Best Params: ", clf.best_params_)


# In[50]:


tfidf_y_predict = clf.predict(X_test)
submission_tfidf = pd.DataFrame({'id':test['id'], "sentiment":tfidf_y_predict})
submission_tfidf.to_csv("./IMDB_tfidf.csv",index=False)


# In[ ]:





# In[50]:


def review_to_texts (raw_text, if_stopwords=False):
    review_text = BeautifulSoup(raw_text).get_text()
    letters_only = re.sub('[^a-zA-Z]'," ",review_text)
    words = letters_only.lower().split()
    if if_stopwords: 
        stops = set(stopwords.words('english'))
        words = [w for w in words if not w in stops]
    return (words)


# In[60]:


# ************************接下来用word2vec*************************

#tokenizer =('C:\\Users\\v_zhiqyyang/nltk_data/punkt/english.pickle')
#tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
import nltk.data

tokenizer =nltk.data.load('punkt/english.pickle')

def review_to_sentences(review, tokenizer,if_stopwords=False):
    to_sentence = tokenizer.tokenize(review.strip())
    sentences = []
    for sentence in to_sentence:
        if len(sentence) > 0:
            sentences.append( review_to_texts( sentence, if_stopwords))
    return sentences


# In[61]:





# In[ ]:





# In[34]:


corpora = []
for review in train["review"]:
    corpora += review_to_sentences(review.decode('utf8'), tokenizer)
    
for review in unlabeled["review"]:
    corpora += review_to_sentences(review.decode('utf8'), tokenizer)


# In[35]:



import logging
from gensim.models import word2vec
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',    level=logging.INFO)


# In[36]:


num_features = 300    # Word vector dimensionality                      
min_word_count = 30   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words


# In[37]:





# In[39]:


word2vec_model = word2vec.Word2Vec(corpora,  workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling)


# In[40]:


model_name = "300features_30minwords_10context"
word2vec_model.save(model_name)


# In[41]:


word2vec_model.doesnt_match("man woman child kitchen".split())


# In[42]:


word2vec_model.most_similar("man")


# In[43]:


word2vec_model.most_similar("beautiful")


# In[44]:


word2vec_model["man"]


# In[93]:


import numpy as np

def makeFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,),dtype="float32")
    index2word_set = set(word2vec_model.wv.index2word)
    nwords=0.
    for w in words:
        if w in index2word_set:
            nwords +=1
            featureVec = np.add(featureVec, word2vec_model[w])
    featureVec = np.divide(featureVec, nwords)
    return featureVec


# In[ ]:





# In[95]:


def getFeatureVec(reviews, model, num_features):
    reviewFeatureVec = np.zeros((len(reviews),num_features), dtype="float32")
    counter=0
    for review in reviews:
        reviewFeatureVec[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
    return reviewFeatureVec



# In[111]:


clean_train_review = []
for review in train["review"]:
    clean_train_review.append(review_to_texts(review.decode('utf8'), if_stopwords=True))
trainVec = getFeatureVec(clean_train_review, word2vec_model, num_features)          

clean_test_review = []
for review in test["review"]:
    clean_test_review.append(review_to_texts(review.decode('utf8'), if_stopwords=True))
testVec = getFeatureVec(clean_test_review, word2vec_model, num_features)   
    


# In[1]:





# In[113]:


y_train = train["sentiment"]
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
gbc = GradientBoostingClassifier(n_estimators=500,learning_rate=0.1,max_depth=4)

gbc.fit(trainVec, y_train)
result = gbc.predict( testVec )


# In[114]:



submission_word2vec = pd.DataFrame({'id':test['id'], "sentiment":result})
submission_word2vec.to_csv("./IMDB_word2vec.csv",index=False)


# In[ ]:




