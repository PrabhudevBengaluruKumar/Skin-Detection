#!/usr/bin/env python
# coding: utf-8

# 
# # Skin Detection Project
# 
# You are given the following csv files (the separator is a ```;```):
# 
# ```
# Project_files/data/2016material-fake.csv
# Project_files/data/2016material.csv
# Project_files/data/2016skin.csv
# Project_files/data/Fleisch.csv
# Project_files/data/Holz.csv
# Project_files/data/Leder.csv
# Project_files/data/Stoff.csv
# Project_files/data/Referenz-Haut_6-Klassen.csv
# ```
# 
# They contain materials and their reflectance factor over certain wavelengths. This data was created for a security application where a system should detect skin and distinguish it from non skin.
# 
# The files ``Project_files/data/2016skin.csv`` and ``Project_files/data/Referenz-Haut_6-Klassen.csv`` contain measurements for skin. All other files contain measurements for different materials that are not skin.
# 
# Your task is to train a classifier that can predict skin vs non skin.
# 
# ### Details
# 
# Your report should be a single Jupyter Notebook and include:
# 
# - Cleaning the data
# - Visualize the data in a meaningful way
# - Measure statistical parameters of the data
# - Compare the performance of different classifiers (you can use the ones from sklearn)
# - Evaluate your classifiers in a meaningful way using appropriate metrics (such as memory consumption, time, F1, accuracy etc)
# - Train for two scenarios, one should minimize the chance of false positives (classifying non skin as skin), one should minimize the chance of false negatives (classifying skin as non skin). Visualize the trade-off between false positives and false negatives if applicable.
# 

# In[2]:


import pandas as pd

files = [
    'Project_files/data/2016material-fake.csv',
    'Project_files/data/2016material.csv',
    'Project_files/data/2016skin.csv',
    'Project_files/data/Fleisch.csv',
    'Project_files/data/Holz.csv',
    'Project_files/data/Leder.csv',
    'Project_files/data/Stoff.csv',
    'Project_files/data/Referenz-Haut_6-Klassen.csv'
]


# **Import Statements**

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import time
import tracemalloc
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, log_loss
from sklearn.preprocessing import MinMaxScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc, f1_score, roc_auc_score,recall_score, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
# !pip install imblearn
# !pip install delayed
import imblearn
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')


# **Cleaning the data**

# In the below function raw data is cleaned by replacing comma decimal by point decimal and then eliminating NAN. Then scaling of features is performed, followed by PCA operation to perform dimensionality reduction on number of features. At last, label column is added to the data frame and assigned value 0 if data contains non-skin features or assigned value 1 if data contains skin features.

# In[4]:


def cleaning_raw_data(data_file,name,skin,decimal_separator):
    if decimal_separator == "point_has_decimal_separator":
        data_file = pd.read_csv(data_file,delimiter=';',  dtype="float64")
    elif decimal_separator == "comma_has_decimal_separator":
        data_file = pd.read_csv(data_file,sep=";" , decimal=',', dtype="float64")
        
    #Removing NAN values from the data
    data_file = data_file.dropna().T
    data_file = data_file[1:]
    data_file.columns = data_file.iloc[0]
    
    #Performing scaling of features of the data
    sc = MinMaxScaler()
    data_file = sc.fit_transform(data_file)
    
    #Reducing number of features present in the data to 6 using PCA
    feature_reduction = PCA(n_components = 6)
    feature_reduction.fit(data_file.T)
    final_features = feature_reduction.components_.T
    data_file = pd.DataFrame(final_features)
    
    #Adding a new column called label. If file contains non-skin data then value 0 is assigned and if the file contains skin data then value 1 is assigned to label
    if skin == "fake_skin":
        labels = pd.DataFrame(np.zeros(data_file.shape[0]), columns=["label"])
        data_file['label'] = labels
    elif skin == "real_skin":
        labels = pd.DataFrame(np.ones(data_file.shape[0]), columns=["label"])
        data_file['label'] = labels
    return data_file


# **1. 2016material_fake file data**

# In[5]:


file1_2016material_fake = cleaning_raw_data(files[0],"2016material_fake","fake_skin","point_has_decimal_separator")
print("First 10 rows after cleaning of data")
print("*"*39)
print(file1_2016material_fake.head(10))


# **2. 2016material data file**

# In[6]:


file2_2016material = cleaning_raw_data(files[1],"2016material","fake_skin","point_has_decimal_separator")
print("First 10 rows after cleaning of data")
print("*"*39)
print(file2_2016material.head(10))


# **3. 2016skin data file**

# In[7]:


file3_2016skin = cleaning_raw_data(files[2],"2016skin","real_skin","point_has_decimal_separator")
print("First 10 rows after cleaning of data")
print("*"*39)
print(file3_2016skin.head(10))


# **4. Fleisch data file**

# In[8]:


file4_Fleisch = cleaning_raw_data(files[3],"Fleisch","fake_skin","comma_has_decimal_separator")
print("First 10 rows after cleaning of data")
print("*"*39)
print(file4_Fleisch.head(10))


# **5. Holz data file**

# In[9]:


file5_Holz = df_Holz = cleaning_raw_data(files[4],"Holz","fake_skin","comma_has_decimal_separator")
print("First 10 rows after cleaning of data")
print("*"*39)
print(file5_Holz.head(10))


# **6. Leder data file**

# In[10]:


file6_Leder = cleaning_raw_data(files[5],"Leder","fake_skin","comma_has_decimal_separator")
print("First 10 rows after cleaning of data")
print("*"*39)
print(file6_Leder.head(10))


# **7. Stoff data file**

# In[11]:


file7_Stoff = cleaning_raw_data(files[6],"Stoff","fake_skin","comma_has_decimal_separator")
print("First 10 rows after cleaning of data")
print("*"*39)
print(file7_Stoff.head(10))


# **8. Referenz_Haut_6_Klassen data file**

# In[12]:


file8_Referenz_Haut_6_Klassen = cleaning_raw_data(files[7],"Referenz_Haut_6_Klassen","real_skin","comma_has_decimal_separator")
print("After cleaning of data")
print("*"*39)
print(file8_Referenz_Haut_6_Klassen.head(10))


# **Combining data from all files and splitting it using train_test_split function from sklearn**

# In[13]:


#Concatinating the data
data = pd.concat([file1_2016material_fake,file2_2016material,file3_2016skin,file4_Fleisch,file5_Holz,file6_Leder,file7_Stoff,file8_Referenz_Haut_6_Klassen], axis=0, ignore_index=True)
X = data.iloc[:,0:-1]
y = data[['label']]

#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# **Statistical summary of data** 

# In[14]:


data.describe()


#  **Common evalution function for all classifiers**

# In[15]:


cols=["Classifier Model", "Accuracy", "Precision", "Recall", "F1 Score", "Log Loss", "Time Required", "Memory Used"]
data = pd.DataFrame(columns=cols)
results_data = pd.DataFrame(columns=cols)

def evaluate_classifier(classifier_model_name,xtest,y_test,starttime,stoptime):
    time_required = stoptime - starttime
    pred = classifier_model_name.predict(xtest)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    prob_pred = classifier_model_name.predict_proba(xtest)
    ll = log_loss(y_test, prob_pred)
    return time, precision, recall, acc, f1, ll


# **Construction of classifiers**

# In[16]:


classifiers_used = [
    SVC(kernel="rbf", probability=True),
    RandomForestClassifier(),
    KNeighborsClassifier(n_neighbors=2),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]



def classifier_model(clasifier_model,classifier_name,results_data):
    
    tracemalloc.start()
    start = time.time()
    clasifier_model.fit(X_train, y_train)
    stop = time.time()
    classifier_memory = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()


    print(classifier_name)
    
    time_required, precision, recall, acc, f1, ll = evaluate_classifier(clasifier_model, X_test, y_test, start, stop)
    print("******Evaluation Results******")
    print("Log loss is",ll)
    print("Precision score is ",precision)
    print("Recall score is ",recall)
    print("Accuracy score is ",acc)
    print("F1 score is ",f1)
    print("******")
    print("Confusion Matrix")
    plot_confusion_matrix(clasifier_model, X_test, y_test,normalize='true')
    plt.show()
    print("******")
    print("Roc Curve")
    y_pred_proba = clasifier_model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.xlabel("False Postive Rate")
    plt.ylabel("True Postive Rate")
    plt.legend(loc=4)
    plt.grid()
    plt.show()
   

    
    evaluation_results_data = pd.DataFrame([[classifier_name, acc*100, precision, recall, f1, (stop-start), classifier_memory, ll]], columns=cols)
    results_data = results_data.append(evaluation_results_data)
    return results_data


# **SVM Classifier**

# In[17]:


results_data = classifier_model(classifiers_used[0],"SVM Classifier",results_data)


# **Random Forest Classifier**

# In[18]:


results_data = classifier_model(classifiers_used[1],"Random Forest Classifier",results_data)


# **KNeighborsClassifier**

# In[19]:


results_data = classifier_model(classifiers_used[2],"KNN Classifier",results_data)


# **Decision Tree Classifier**

# In[20]:


results_data = classifier_model(classifiers_used[3],"Decision Tree Classifier",results_data)


# **AdaBoost Classifier**

# In[21]:


results_data = classifier_model(classifiers_used[4],"AdaBoost Classifier",results_data)


# **Gradient Boosting Classifier**

# In[22]:


results_data = classifier_model(classifiers_used[5],"Gradient Boosting Classifier",results_data)


# **Gaussian Naive Bayes Classifier**

# In[23]:


results_data = classifier_model(classifiers_used[6],"Gaussian Naive Bayes Classifier",results_data)


# **Logistic Regression  Classifier**

# In[24]:


results_data = classifier_model(classifiers_used[7],"Logistic Regression Classifier",results_data)


# **Linear Discriminant Analysis Classifier**

# In[25]:


results_data = classifier_model(classifiers_used[8],"Linear Discriminant Analysis Classifier",results_data)


# **Quadratic Discriminant Analysis Classifier**

# In[26]:


results_data = classifier_model(classifiers_used[9],"Quadratic Discriminant Analysis Classifier",results_data)


# **Comparison between classifiers**

# In[27]:


def print_comparison_plot(metric_name,results_data):
    sns.set_color_codes("muted")
    sns.barplot(x=metric_name, y='Classifier Model', data=results_data, color="blue")

    plt.xlabel(metric_name)
    plt.title('Comparison of '+metric_name+' from different classifiers')
    plt.grid()
    plt.show()


# **Accuracy Comparison**

# In[28]:


print_comparison_plot("Accuracy",results_data)


# **Precision Comparison**

# In[29]:


print_comparison_plot("Precision",results_data)


# **Recall Comparison**

# In[30]:


print_comparison_plot("Recall",results_data)


# **F1 Score Comparison**

# In[31]:


print_comparison_plot("F1 Score",results_data)


# **Log Loss Comparison**

# In[32]:


print_comparison_plot("Log Loss",results_data)


# **Time Required Comparison**

# In[33]:


print_comparison_plot("Time Required",results_data)


# **Memory Used Comparison**

# In[34]:


print_comparison_plot("Memory Used",results_data)


# **Sampling**

# To minimize the chance of false positives (classifying non skin as skin) or to minimize the chance of false negatives (classifying skin as non skin), sampling of data is done.

# In[35]:


data_for_sampling = pd.concat([file1_2016material_fake,file2_2016material,file3_2016skin,file4_Fleisch,file5_Holz,file6_Leder,file7_Stoff,file8_Referenz_Haut_6_Klassen], axis=0, ignore_index=True)
X = data_for_sampling.iloc[:,0:-1]
y = data_for_sampling[['label']]

print("Visualization of data before sampling")
sns.countplot('label', data=data_for_sampling)
plt.title('Before Sampling')
plt.xticks(np.arange(len(('non skin : 0','skin : 1'))), ('non skin : 0','skin : 1'))
plt.grid()
plt.show()


#Over-Sampling is done here with SMOTE function using imblearn
print("Visualization of data after Over Sampling")
oversampling = SMOTE()
X_oversample_data, y_oversample_data = oversampling.fit_resample(X, y)
oversampled_data = pd.concat([X_oversample_data, y_oversample_data])
sns.countplot('label', data=oversampled_data)
plt.title('After Over Sampling')
plt.xticks(np.arange(len(('non skin : 0','skin : 1'))), ('non skin : 0','skin : 1'))
plt.grid()
plt.show()


#Under-Sampling is done here
print("Visualization of data after Under Sampling")
class_1,class_2 = data_for_sampling.label.value_counts()
c1 = data_for_sampling[data_for_sampling['label'] == 0]
c2 = data_for_sampling[data_for_sampling['label'] == 1]
df_3 = c2.sample(class_2)
df_2 = c1.sample(class_2)
undersampled_data = pd.concat([df_3, df_2])
sns.countplot('label', data=undersampled_data)
plt.title('After Under Sampling')
plt.xticks(np.arange(len(('non skin : 0','skin : 1'))), ('non skin : 0','skin : 1'))
plt.grid()
plt.show()


print("*"*39)
print("Over-Sampling data containing NAN values")
print(oversampled_data)
oversampled_data = oversampled_data.fillna(0)
print("*"*39)
print("Replacing NAN with 0 after Over-Sampling")
print(oversampled_data)


# In[36]:


X_oversample = oversampled_data.iloc[:,0:-1]
y_oversample = oversampled_data[['label']]

#Splitting the data
X_train_oversample, X_test_oversample, y_train_oversample, y_test_oversample = train_test_split(X_oversample, y_oversample, test_size=0.25, random_state=42)

X_undersample = undersampled_data.iloc[:,0:-1]
y_undersample = undersampled_data[['label']]

X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample, y_undersample, test_size=0.25, random_state=42)


# In[37]:


cols2=["Classifier Model", "Accuracy", "Precision", "Recall", "F1 Score", "Log Loss", "Time Required", "Memory Used"]
oversam_data = pd.DataFrame(columns=cols2)
oversam_results_data = pd.DataFrame(columns=cols2)

undersam_data = pd.DataFrame(columns=cols2)
undersam_results_data = pd.DataFrame(columns=cols2)

def evaluate_classifier(classifier_model_name,xtest,y_test,starttime,stoptime):
    time_required = stoptime - starttime
    pred = classifier_model_name.predict(xtest)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    prob_pred = classifier_model_name.predict_proba(xtest)
    ll = log_loss(y_test, prob_pred)
    return time, precision, recall, acc, f1, ll


# In[38]:


classifiers_used = [
    SVC(kernel="rbf", probability=True),
    RandomForestClassifier(),
    KNeighborsClassifier(n_neighbors=2),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LogisticRegression(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]



def classifier_model_sampling(clasifier_model,classifier_name,oversam_results_data,undersam_results_data):
    
    tracemalloc.start()
    start = time.time()
    clasifier_model.fit(X_train, y_train)
    stop = time.time()
    classifier_memory = tracemalloc.get_tracemalloc_memory()
    tracemalloc.stop()


    print(classifier_name)
    
    time_required, precision, recall, acc, f1, ll = evaluate_classifier(clasifier_model, X_test_oversample, y_test_oversample, start, stop)
    print("******Over-Sampling Evaluation Results******")
    print("Log loss is",ll)
    print("Precision score is ",precision)
    print("Recall score is ",recall)
    print("Accuracy score is ",acc)
    print("F1 score is ",f1)
    print("******")
    print("Confusion Matrix")
    plot_confusion_matrix(clasifier_model, X_test, y_test,normalize='true')
    plt.show()
    
    oversam_evaluation_results_data = pd.DataFrame([[classifier_name, acc*100, precision, recall, f1, (stop-start), classifier_memory, ll]], columns=cols)
    oversam_results_data = oversam_results_data.append(oversam_evaluation_results_data)
    
    time_required, precision, recall, acc, f1, ll = evaluate_classifier(clasifier_model, X_test_undersample, y_test_undersample, start, stop)
    print("******Under-Sampling Evaluation Results******")
    print("Log loss is",ll)
    print("Precision score is ",precision)
    print("Recall score is ",recall)
    print("Accuracy score is ",acc)
    print("F1 score is ",f1)
    print("******")
    print("Confusion Matrix")
    plot_confusion_matrix(clasifier_model, X_test, y_test,normalize='true')
    plt.show()
    
    undersam_evaluation_results_data = pd.DataFrame([[classifier_name, acc*100, precision, recall, f1, (stop-start), classifier_memory, ll]], columns=cols)
    undersam_results_data = undersam_results_data.append(undersam_evaluation_results_data)
    
    return oversam_results_data,undersam_results_data

#     return results_data


# **SVM Classifier after Over-Sampling and Under_Sampling**

# In[39]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[0],"SVM Classifier",oversam_results_data,undersam_results_data)


# **Random Forest Classifier after Over-Sampling and Under_Sampling**

# In[40]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[1],"Random Forest Classifier",oversam_results_data,undersam_results_data)


# **KNN Classifier after Over-Sampling and Under_Sampling**

# In[41]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[2],"KNN Classifier",oversam_results_data,undersam_results_data)


# **Decision Tree Classifier after Over-Sampling and Under_Sampling**

# In[42]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[3],"Decision Tree Classifier",oversam_results_data,undersam_results_data)


# **AdaBoost Classifier after Over-Sampling and Under_Sampling**

# In[43]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[4],"AdaBoost Classifier",oversam_results_data,undersam_results_data)


# **Gradient Boosting Classifier after Over-Sampling and Under_Sampling**

# In[44]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[5],"Gradient Boosting Classifier",oversam_results_data,undersam_results_data)


# **GaussianNB Classifier after Over-Sampling and Under_Sampling**

# In[45]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[6],"GaussianNB Classifier",oversam_results_data,undersam_results_data)


# **Logistic Regression Classifier after Over-Sampling and Under_Sampling**

# In[46]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[6],"Logistic Regression Classifier",oversam_results_data,undersam_results_data)


# **Linear Discriminant Analysis Classifier after Over-Sampling and Under_Sampling**

# In[47]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[6],"Linear Discriminant Analysis Classifier",oversam_results_data,undersam_results_data)


# **Quadratic Discriminant Analysis Classifier after Over-Sampling and Under_Sampling**

# In[48]:


oversam_results_data,undersam_results_data = classifier_model_sampling(classifiers_used[6],"Quadratic Discriminant Analysis Classifier",oversam_results_data,undersam_results_data)


# **Comparison between Over-Sampling and Under-Sampling of data to show trade-off between false positives and false negatives**

# In[49]:


def print_comparison_plot2(metric_name,oversam_results_data,undersam_results_data):
    print("Over-Sampling")
    print("*"*39)
    sns.set_color_codes("muted")
    sns.barplot(x=metric_name, y='Classifier Model', data=oversam_results_data, color="blue")

    plt.xlabel(metric_name)
    plt.title('Comparison of '+metric_name+' from different classifiers')
    plt.grid()
    plt.show()
    
    
    print("Under-Sampling")
    print("*"*39)
    sns.set_color_codes("muted")
    sns.barplot(x=metric_name, y='Classifier Model', data=undersam_results_data, color="blue")

    plt.xlabel(metric_name)
    plt.title('Comparison of '+metric_name+' from different classifiers')
    plt.grid()
    plt.show()


# **Accuracy comparison after Over-Sampling and Under-Sampling**

# In[50]:


print_comparison_plot2("Accuracy",oversam_results_data,undersam_results_data)


# **Precision comparison after Over-Sampling and Under-Sampling**

# In[51]:


print_comparison_plot2("Precision",oversam_results_data,undersam_results_data)


# **Recall comparison after Over-Sampling and Under-Sampling**

# In[52]:


print_comparison_plot2("Recall",oversam_results_data,undersam_results_data)


# **F1 Score comparison after Over-Sampling and Under-Sampling**

# In[53]:


print_comparison_plot2("F1 Score",oversam_results_data,undersam_results_data)


# **Log Loss comparison after Over-Sampling and Under-Sampling**

# In[54]:


print_comparison_plot2("Log Loss",oversam_results_data,undersam_results_data)


# **Time Required comparison after Over-Sampling and Under-Sampling**

# In[55]:


print_comparison_plot2("Time Required",oversam_results_data,undersam_results_data)


# **Memory Used comparison after Over-Sampling and Under-Sampling**

# In[56]:


print_comparison_plot2("Memory Used",oversam_results_data,undersam_results_data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




