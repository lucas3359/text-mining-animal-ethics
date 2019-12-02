#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import KFold
import numpy as np
import string
import re
from collections import Counter
from nltk.corpus import stopwords
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim.models
from gensim.models import Word2Vec
from gensim.models import Doc2Vec


# In[63]:


train = pd.read_csv('J:/Lucas/Animal Research/query/Liggins faculty/Liggins_exclude2011.csv',encoding = "ISO-8859-1")

train


# In[64]:


# text cleaning
def clean_text(text):
    clean1 = re.sub(r'['+string.punctuation + '’—”'+']', "", text.lower())
    return re.sub(r'\W+', ' ', clean1)

train['tokenized'] = train['Text'].map(lambda x: clean_text(x))
train['tokenized']


# In[65]:


train['uniq_wds'] = train['tokenized'].str.split().apply(lambda x: len(set(x)))
train['uniq_wds'].head(10)


# In[11]:


train


# In[66]:


stop = stopwords.words('english')
train['tokenized'] = train['tokenized'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[67]:


train['tokenized'].head()


# In[68]:


freq = pd.Series(' '.join(train['tokenized']).split()).value_counts()[:10]


# In[69]:


freq


# In[70]:


train['tokenized'][1]


# In[72]:


train_x=train['tokenized']
train_y =train['animal_related']


# In[73]:


train_x


# In[74]:


train_y


# In[22]:


count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train['tokenized'])


# In[75]:


test = pd.read_csv('J:/Lucas/Animal Research/query/Liggins faculty/Liggins2011.csv',encoding = "ISO-8859-1")
test


# In[76]:


test_x=test['Text']
test_y =test['animal_related']


# In[77]:


xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(test_x)


# In[78]:


test_x


# In[31]:


xtrain_count


# In[32]:


xvalid_count


# In[79]:


# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(train['tokenized'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(test_x)

# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=5000)
tfidf_vect_ngram.fit(train['tokenized'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(test_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(1,3), max_features=5000)
tfidf_vect_ngram_chars.fit(train['tokenized'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(test_x) 


# In[91]:


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, test_y)


# In[81]:


accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print("NB, Count Vectors: ", accuracy)


# In[82]:


accuracy2 = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print("NB, WordLevel TF-IDF: ", accuracy2)


# In[83]:


accuracy3 = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("NB, N-Gram Vectors: ", accuracy3)


# In[84]:


accuracy4 = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print("NB, CharLevel Vectors: ", accuracy4)


# In[85]:


accuracy = train_model(linear_model.LogisticRegression(solver='lbfgs', multi_class='auto'), xtrain_count, train_y, xvalid_count)
print ("LR, Count Vectors: ", accuracy)

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(solver='lbfgs', multi_class='auto'), xtrain_tfidf, train_y, xvalid_tfidf)
print ("LR, WordLevel TF-IDF: ", accuracy)

# Linear Classifier on Ngram Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(solver='lbfgs', multi_class='auto'), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("LR, N-Gram Vectors: ", accuracy)

# Linear Classifier on Character Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(solver='lbfgs', multi_class='auto'), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print ("LR, CharLevel Vectors: ", accuracy)


# In[86]:


accuracy = train_model(svm.SVC(gamma="scale"), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print ("SVM, N-Gram Vectors: ", accuracy)


# In[41]:


from sklearn.ensemble import RandomForestRegressor

# RF on Count Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print ("RF, Count Vectors: ", accuracy)

# RF on Word Level TF IDF Vectors
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)
print ("RF, WordLevel TF-IDF: ", accuracy)


# In[87]:


# Extereme Gradient Boosting on Count Vectors
import xgboost
accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print ("Xgb, Count Vectors: ", accuracy)

# Extereme Gradient Boosting on Word Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())
print ("Xgb, WordLevel TF-IDF: ", accuracy)

# Extereme Gradient Boosting on Character Level TF IDF Vectors
accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())
print ("Xgb, CharLevel Vectors: ", accuracy)


# In[185]:


####################### Neural Network
import tensorflow as tf
from tensorflow import keras
def create_model_architecture(input_size):
    # create input layer 
    input_layer = keras.layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = keras.layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = keras.layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 

classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])
accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)
print ("NN, Ngram Level TF IDF Vectors",  accuracy)


# In[90]:


def train_model2(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    return predictions


# In[92]:


from sklearn import svm
prediction = train_model2(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print ("prediction: ", prediction)






# In[93]:


prediction = train_model2(linear_model.LogisticRegression(solver='lbfgs', multi_class='auto'), xtrain_count, train_y, xvalid_count)
print ("prediction: ", prediction)


# In[94]:


prediction = train_model2(linear_model.LogisticRegression(solver='lbfgs', multi_class='auto'), xtrain_tfidf, train_y, xvalid_tfidf)
print ("prediction: ", prediction)


# In[95]:


prediction = train_model2(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print("prediction: ", prediction)


# In[96]:


test_y


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[241]:





# In[ ]:





# In[194]:





# In[236]:





# In[ ]:


#####Word2Vec


# In[245]:



from nltk.tokenize import RegexpTokenizer

data_list = list()
indv_lines = train['tokenized'].values.tolist()
for line in indv_lines:
    rem_tok_punc = RegexpTokenizer(r'\w+')
    tokens = rem_tok_punc.tokenize(line)
    words = [w.lower() for w in tokens]
    
    data_list.append(words)
len(data_list)


# In[248]:


import gensim
Embedding_Dim = 100
model = gensim.models.Word2Vec(sentences = data_list, size = Embedding_Dim,workers = 4,min_count = 1)
words = list(model.wv.vocab)
print('%d' % len(words))


# In[249]:


model.wv.most_similar('land')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[227]:


model = Word2Vec(Word)


# In[228]:


print(model)


# In[229]:


words = list(model)
print(words)


# In[203]:


model['land']


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


########################################################################################################


# In[7]:


wd_counts = Counter()
for i, row in train.iterrows():
    wd_counts.update(row['tokenized'].split())


# In[10]:


wd_counts.most_common(20)


# In[11]:


from nltk.corpus import stopwords
stopwords.words('english')

for sw in stopwords.words('english'):
    del wd_counts[sw]
    
wd_counts
wd_counts.most_common(20)


# In[18]:


Theme_group =train.groupby('Theme')
Theme_group
Theme_group['text'].count()
Theme_proportions = Theme_group["text"].count()/train["text"].count()
Theme_proportions


# In[34]:


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train['text'], train['Theme'])


# In[24]:


train_x


# In[27]:


count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(train['text'])


# In[43]:


xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


# In[31]:


help(preprocessing.LabelEncoder)


# In[42]:


encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


# In[44]:


xtrain_count


# In[46]:





# In[47]:


xtrain_tfidf


# In[61]:


model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[50]:


embeddings_index = {}
for i, line in enumerate(open('GoogleNews-vectors-negative300.bin')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(train['text'])
word_index = token.word_index

# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


# In[ ]:




