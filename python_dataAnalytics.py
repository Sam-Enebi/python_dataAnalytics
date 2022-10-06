#!/usr/bin/env python
# coding: utf-8

"""
Created on a sunny day...

@Student name : Sam Enebi 
@Student ID: R00167276
@Cohort: SD3
"""

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# In[11]:

def Task1():
    # Your implemention for Task 1...
    df = pd.read_csv('weatherAUS.csv')
    return df.groupby('Location').agg(Avg_Rainfall = ('Rainfall','mean')).sort_values(by = 'Avg_Rainfall', ascending = False)

Task1()

# In[13]:

def Task2():
    # Your implemention for Task 2...
    df = pd.read_csv('weatherAUS.csv')
    maximum_temperature = df.groupby('Location').agg(Maximum_Temp = ('MaxTemp','max')).sort_values(by = 'Maximum_Temp', ascending = False)
    
    maximum_temperature = maximum_temperature[maximum_temperature['Maximum_Temp'] > 46.6]
    print(maximum_temperature)
    maximum_temperature.plot(kind = 'line')
    plt.xticks(rotation=45)  
    plt.ylabel('Temperature')
    plt.show()
Task2()

# In[16]:

def Task3():
    # Your implemention for Task 3...
    df = pd.read_csv('weatherAUS.csv')
    df['RainTomorrow'] = df['RainTomorrow'].map({'No':0, 'Yes':1})
    #drop any null values if any
    df.dropna(subset = ['WindSpeed9am', 'Humidity9am', 'Pressure9am', 'RainTomorrow', 'WindSpeed3pm', 'Humidity3pm', 'Pressure3pm'],  inplace = True)
    
    #data1
    set1 = df[['WindSpeed9am', 'Humidity9am', 'Pressure9am', 'RainTomorrow']]
    
    #creating x and y 
    X = set1.loc[:,set1.columns!='RainTomorrow']
    y = set1['RainTomorrow']

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(x_train,y_train)
    dt_y_pred = dt_classifier.predict(x_test)
    print("Decision Tree Classifier first Dataset's score: " , dt_classifier.score(x_test,y_test))

    #data2

    set2 = df[['WindSpeed3pm', 'Humidity3pm', 'Pressure3pm', 'RainTomorrow']]

    #creating x and y 

    X = set2.loc[:,set1.columns!='RainTomorrow']
    y = set2['RainTomorrow']

    #spliting the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    dt_classifier = DecisionTreeClassifier()
    dt_classifier.fit(x_train,y_train)
    dt_y_pred = dt_classifier.predict(x_test)
    print("Decision Tree Classifier second Dataset's score: " , dt_classifier.score(x_test,y_test))
        
Task3()

# In[6]:

def Task4():
    # Your implemention for Task 4...
    df = pd.read_csv('weatherAUS.csv', usecols = ['MaxTemp', 'MinTemp', 'WindGustSpeed', 'Rainfall', 'RainTomorrow'])
    df['RainTomorrow'] = df['RainTomorrow'].map({'No':0, 'Yes':1})
    df.dropna(inplace = True)
    columns = ['MaxTemp', 'MinTemp', 'WindGustSpeed', 'Rainfall']
    accuracy = {}
    for column in columns:
        data = df[[column, 'RainTomorrow']]
        X = data.loc[:,data.columns!='RainTomorrow']
        y = data['RainTomorrow']
        #spliting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
        #Our next step is to K-NN model and train it with the training data. n_neighbors is the value of factor K.
        knn_classifier = KNeighborsClassifier(n_neighbors = 5)
        knn_classifier.fit(X_train,y_train)
        #training the data and testing the accuracy
        y_pred_knn = knn_classifier.predict(X_test)
        score = accuracy_score(y_test,y_pred_knn)
        print(f'{column} Column Accuracy is :',score)
        accuracy[column] = score
    print(f"Maximum Accuracy is: {accuracy[max(accuracy, key=accuracy.get)]}")
    return df

Task4()

# In[10]:

def Task5():
    df = pd.read_csv('weatherAUS.csv', usecols = ['WindGustDir', 'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm'])
    df.dropna(inplace = True)
    df['WindGustDir'] = df['WindGustDir'].astype('category')
    df['WindGustDir'] = df['WindGustDir'].cat.codes
    X = df.values

    # Using the elbow method to find the optimal number of clusters
    wcss = [] 
    for i in range(1, 11): 
        k_means = KMeans(n_clusters = i, init = 'k-means++', random_state = 40)
        k_means.fit(X) 
        wcss.append(k_means.inertia_)

    dic_inertia = dict(zip(range(1,11), wcss))
    print("inertia scores (sum of squared errors) by number of clusters:")
    _ = [print(k, ":", f'{v:,.0f}') for k,v in dic_inertia.items()]

    # plot
    plt.plot(range(1, 11), wcss) 
    plt.title('The Elbow Method showing the optimal k')
    plt.xlabel('Number Of Clusters') 
    plt.ylabel('Distortion')
    plt.show()
    
    #The elbow shape is created at point 4, that is, our K value or an optimal number of clusters is 4.
    
    k_means = KMeans(n_clusters = 4, init = "k-means++", random_state = 40)
    y_kmeans = k_means.fit_predict(X)
    
Task5()




