'''
Group : Avocado
Nama Anggota Group :
- Vivian Davina Hendrawan       19.K1.0011
- Selly Margaretha Sudiyandi    19.K1.0046
- Ninda Setyowati               19.K1.0059
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk

from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA

# Supaya data tidak teracak
np.random.seed(100)

# Load csv
dataset = pd.read_csv('data/sms.csv', delimiter=',')
dataset.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
y = dataset[['v1']]
#print(dataset.head())
print("Jumlah Masing-masing cluster : ")
print(dataset['v1'].value_counts())

# Untuk memberi label pada spam dan ham
def converty(row):
    if row == 'spam':
        return 1
    else:
        return 0
dataset['ytrue']=dataset['v1'].apply(converty)

# Text cleaning + normalization
def cleaned(text):
    stopword = stopwords.words('english')
    lower = text.lower()
    text = re.sub(r'[0-9]', '', lower)
    text = re.sub(r'ã¼', '', text)
    text = re.sub(r'â', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = nltk.tokenize.word_tokenize(text)
    keywords = [keyword for keyword in tokens if keyword.isalpha() and not keyword in stopword]
    return ' '.join(keywords)
dataset['cleaned'] = dataset['v2'].apply(cleaned)
print("\nCleaning Dataset : ")
print(dataset['cleaned'])

#Menghitung TF-IDF
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(dataset['cleaned'])
print("\nTF-IDF : ")
print(x)

# Menghitung K-Means
kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
print("\nK-Means : ")
print(kmeans)

clusters = kmeans.labels_.tolist()
dataset['hasil'] = clusters
print(dataset.head(20))
#print(clusters)

pca = PCA(2)
data = pca.fit_transform(x.toarray())
centers = np.array(kmeans.cluster_centers_)
model = KMeans(n_clusters = 2, init = "k-means++")
label = model.fit_predict(x)
plt.figure(figsize=(10,10))
uniq = np.unique(label)
for i in uniq:
   plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
plt.scatter(centers[:,0], centers[:,1], marker="x", color='k')
plt.legend()
plt.show()

# Test Data
dataset2 = pd.read_csv('data/test.csv', delimiter=',')
dataset2['cleaned'] = dataset2['v2'].apply(cleaned)
dataset2.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
x2 = vectorizer.transform(dataset2['cleaned'])
y2 = kmeans.predict(x2)
dataset2['hasilprediksi'] = y2
print("\nHasil Prediksi : ")
print(dataset2)

print("\nAccuracy : ", f1_score(dataset[['ytrue']],dataset[['hasil']]))