import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#Import dataset
df = pd.read_csv('C:/Users/ACER/PycharmProjects/KNN/dataset.csv');

#Data Cleaning
df.dropna()
data = df.loc[df.iloc[:,3]>=0]

#Clustering using K Means
km = KMeans(n_clusters=3)
predicted = km.fit_predict(data.iloc[:,[3]])
data['Cluster'] = predicted

#Split train test
X = data.drop(['Cluster','tahun','wilayah','penyebab'], axis =1)
y = data['Cluster']
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size= 0.2, random_state=5)

#Classify using KNN
knn = KNeighborsClassifier(n_neighbors=3)

#Learning
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print(f"Accuracy : {np.mean(y_pred == y_test)}")

