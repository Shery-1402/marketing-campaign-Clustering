#!/usr/bin/env python
# coding: utf-8

# #**Data Science - Machine Learning Project**
# 
# ##**Identify targeted group of customers**

# ####**Introduction**
# 
# Clustering involves grouping data objects into several clusters where objects in the same cluster are similar to each other and dissimilar to those in other clusters. This process is based on the attributes of the objects, often using distance measures to assess similarities and dissimilarities. In this task, based on the business problem two types of Clustering methods will be implanted on to the dataset. 
# 
# #####**Business Problem :***
# 
# The project's main goal is to analyze data on a company's ideal customers to identify specific customer groups. This understanding will enable the business to tailor products to the unique needs, behaviors, and concerns of different customer segments.
# 
# In today's market, consumers seek variety in their purchases, whether online or in-store. Therefore, analyzing customer personalities is crucial for businesses to adjust their offerings to meet the demands of various customer segments. For instance, instead of marketing a new product to all customers, the company can focus on the segment most likely to be interested, optimizing marketing efforts and resources.
# 
# 
# 
# #####**Data**
# Considering the nature of our problem, the main data required will be the Customer personality dataset. The data is collected from Kaggle for clustering. 
# 
# 
# 1. [Customer Personality Data](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
#     
#     The dataset consits of both Customer demographics data as well as purchase details, from a business registered between 2012 to 2014.  We wil be looking at all data, by their variety of attibutes such as education background marital status, etc. to identify clusters of consumers so we will be pre-processing the data to get the relevant records for profiling. Data below is devided into sections such as people demographics, product types, Promotions, & Source of purchase.  
#     
#     
#     
#     Data Attributes :
#  
#       - **People**
# 
# - ID: Customer's unique identifier
# - Year_Birth: Customer's birth year
# - Education: Customer's education level
# - Marital_Status: Customer's marital status
# - Income: Customer's yearly household income
# - Kidhome: Number of children in customer's household
# - Teenhome: Number of teenagers in customer's household
# - Dt_Customer: Date of customer's enrollment with the company
# - Recency: Number of days since customer's last purchase
# - Complain: 1 if the customer complained in the last 2 years, 0 otherwise.
# 
#     - **Products**
# 
# - MntWines: Amount spent on wine in last 2 years
# - MntFruits: Amount spent on fruits in last 2 years
# - MntMeatProducts: Amount spent on meat in last 2 years
# - MntFishProducts: Amount spent on fish in last 2 years
# - MntSweetProducts: Amount spent on sweets in last 2 years
# - MntGoldProds: Amount spent on gold in last 2 years
# 
#     - **Promotion**
#     
# - NumDealsPurchases: Number of purchases made with a discount
# - AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# - AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# - AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# - AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# - AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# - Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# 
# 
#     - **Source of purchase**
# 
# - NumWebPurchases: Number of purchases made through the company’s website
# - NumCatalogPurchases: Number of purchases made using a catalogue
# - NumStorePurchases: Number of purchases made directly in stores
# - NumWebVisitsMonth: Number of visits to company’s website in the last month
# 
#   
#   Couple of things we could analyse the data by:
#   
#   - Total spendings by Education - does the data have any correlation between education and spendings?
#   - Total spendings by Marital status - analysis between consumers who married and not and alone.
#   - Find couples spending behaviours
#   - Does age have any impact on the amount of money they spend and income?
# 

# # **Importing Required Libraries**

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().system('pip install scikit-learn')

print('Libraries imported.')


# # **Importing the Data**

# In[3]:


marketing_campaign_dataset = pd.read_csv('marketing_campaign.csv')
marketing_campaign_dataset


# #### **Analysis of the Data**
# 
# Using pandas library we have loaded the data into a dataframe, called as '**marketing_campaign_dataset**'. 
# 
# You can see from the above that the data has 29 attributes with 2240 rows of data. 2 of the attributes (Education and Marital Status are categorical, and others are Numerical. 
# 
# 

# In[3]:


marketing_campaign_dataset.head()


# In[4]:


marketing_campaign_dataset.tail()


# #  **Data Cleaning**

# In[4]:


marketing_campaign_dataset.info()


# In[5]:


marketing_campaign_dataset.isnull().sum()


# ### **Analysis of the Data**
# 
# Info() function in python gives us a concise summary of the data including if there are any missing values if so which column and understand the type of data in each column. 
# 
# As we can see here, all of the columns has 2240 Non-Null values except Income which only has 2216. Therefore we have **24** missing values in the Income column.
# 
# 
# So lets look further into this Income column. 
# 

# In[6]:


marketing_campaign_dataset.describe()
# Min incomes in the dataset is 1730 where as max value is 666666.


# In[7]:


marketing_campaign_dataset['Income'].plot(kind='box')


# #### Analysis of the Boxplot visual
# 
# Looking at the Income value distribution using box visual, 
# we can see that vast majority of the customers income are clustered between 0 - 100,000, 
# which represents by the box, but there are a few with exceptionally high incomes that potentiall can be outliers. 
# especially with over 600,000 annual income. 
# 

# #### We can further analyse this Income data by using the scatter plot visual since its a numerical values and see if there are any relationship between Year_Birth and Income values. It will show us not only report the values of the individual data points, but also patterns when the data are taken as a whole.  However, to get the actual Age of each customer we would have to calculate the age based on this attribute - currently calender year. 
# 

# # Age outlier check

# In[8]:


#Age of customer as of this year 2023 
marketing_campaign_dataset["Age"] = 2023-marketing_campaign_dataset["Year_Birth"]
marketing_campaign_dataset.head()


# In[9]:


pl = sns.scatterplot(x=marketing_campaign_dataset["Income"], y=marketing_campaign_dataset["Age"],
                     data = marketing_campaign_dataset)
pl.set_title("Income versus Age")
plt.legend()
plt.show()


# In[10]:


marketing_campaign_dataset['Age'].plot(kind='box')
# Visualising the Age in box plot to identify the outliers


# In[11]:


marketing_campaign_dataset.describe()


# In[14]:


marketing_campaign_dataset=marketing_campaign_dataset[marketing_campaign_dataset['Age']<=91]


# In[15]:


marketing_campaign_dataset['Age'].plot(kind='box')
# Visualising the Age in box plot to identify the outliers


# In[19]:


sns.histplot(marketing_campaign_dataset.Age)
plt.title('Age Distribution')
plt.show()


# #### Analysis of the Bar visual
# 
# As we took the outliers from the birthyear attribute, we can see now the age is now more looking like normally distributed. Consumers in the dataset looks to be is in-between 20 - 90 years of old, where majority of the consumers are around 45-55 years of old. 

# # Salary outlier check 

# In[16]:


Potential_Outlier=marketing_campaign_dataset[marketing_campaign_dataset['Income']>200000]
Potential_Outlier


# In[ ]:


#Looking specifically at the customer with 666,666 annual salary there is only record.

#There are many way to find the outliers, one of which is find the IQR. 
#IQR is the range between the first quartile (25th percentile) and the third quartile (75th percentile). 
#Any data point that lies below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR can be considered an outlier.

#Based on this dataset, the Q1 = 35303.000000, Q3 = 68522.000000, Therefore IQR = Q3-Q1 = 33219
#So, based on the formula, Q3+1.5*IQR = 68522+1.5*33219 = 118,351
#Therefore we can conclude the above record is an outlier. 


# # excluded income over 200,000

# In[17]:


marketing_campaign_dataset=marketing_campaign_dataset[marketing_campaign_dataset['Income']<=118351]
marketing_campaign_dataset['Income'].max()

# Excluding consumers with over 200,000 annual income from the dataset. 


# In[18]:


marketing_campaign_dataset['Income'].plot(kind='box')


# In[105]:


pl = sns.scatterplot(x=marketing_campaign_dataset["Income"], y=marketing_campaign_dataset["Age"],
                     data = marketing_campaign_dataset)
pl.set_title("Income versus Year_Birth")
plt.legend()
plt.show()


# #### Analysis of Scatter plot visual
# 
# It is challenging to discern a clear trend here due to the density of the data, as there diesnt appear to be a strong correlation between the birth year and how much imcome  they receive. Most of the variation in Income is within the same birth year groups. 
# 
# It is evident that significant amount of data point is observer between 1940 and 2000 year with income mostly in between 0 and 100,000. 
# 
# 
# From the Scatter plot above, one fact is obvious that a single point on the far right indicates an individual with an income that significantly surpassess others in the dataset. 
# 
# It is worth noting that, there is a region with fewer data points, potentially outliers,  by deviating significantly away from the dence cluster in terms of the birth year this shows that in the dataset we have some customers who are above 100 years old with almost 100,000 income. Potentially we can say that both this indivisual customer with over 600,000 and consumer over 100 years old are outliers.
# 
# Before we do anything with the Year_Birth attribute, we can add in a new attribute derived from the Year_Birth column to calculate the consumers age for further analysis in box plot.
# 

# # Adding new attribute (total spent) into the dataset and dropping rest unnecessary attributes

# In[107]:


marketing_campaign_dataset["totalspent"] = marketing_campaign_dataset["MntWines"]+ 
                                            marketing_campaign_dataset["MntFruits"]+
                                            marketing_campaign_dataset["MntMeatProducts"]+
                                            marketing_campaign_dataset["MntFishProducts"]+ 
                                            marketing_campaign_dataset["MntSweetProducts"]+ 
                                            marketing_campaign_dataset["MntGoldProds"]

#Dropping some of the redundant features
to_drop = ["IsParent", "Total_Children", "Marital_Status", "Education", "Kidhome", "Teenhome",
            "Dt_Customer", "Recency","MntWines", "MntFruits", "MntMeatProducts",
           "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", 
           "NumStorePurchases","NumWebVisitsMonth", "Z_CostContact","Z_Revenue",
           "Year_Birth", "ID", "Recency","Total_Children", "Complain", "Response",
           "AcceptedCmp5", "AcceptedCmp4", "AcceptedCmp3", "AcceptedCmp2", "AcceptedCmp1"]
final_dataset = marketing_campaign_dataset.drop(to_drop, axis=1)

final_dataset


# In[108]:


sns.histplot(marketing_campaign_dataset.totalspent)
plt.title('Spent Distribution')
plt.show()


# #### Analysis of the Bar visual
# 
# As we took the outliers from the Spent attribute, we can see now the Spent attribute is more right skwed where majority of the consumers spend their money between 0-200.

# In[110]:


final_dataset.info()


# In[123]:


sns.pairplot(final_dataset.iloc[:,[0,1,2]])


# # Standardize the data

# In[124]:


from sklearn.preprocessing import StandardScaler
X= final_dataset.iloc[:, [0,2]].values
sc_X = StandardScaler()
X=sc_X.fit_transform(X)


# # Choose the number of clusters (k) using the elbow method

# In[125]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # # K-means Clustering - Part 1

# In[126]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X) # Fit the model
y_kmeans


# In[127]:


plt.figure(figsize=(15,8))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 0')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 3')
 #plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'black', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers using K-means')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.legend()
plt.show()


# Analysis
# 
# 1. Customers in Cluster 0 (Red) Customers with moderate income and moderate spending.
# 2. Customers in Cluster 1 (Blue) Customers with high income and high spending.
# 3. Customers in Cluster 2 (Green) Customers with low income and low spending.
# 4. Customers in Cluster 3 (Cyan) Customers with moderate income and very high spending.

# In[142]:


final_dataset['Cluster'] = y_kmeans 

cluster_agg = final_dataset.groupby('Cluster').agg({
    'totalspent': ['sum', 'count'],  
    'Income': ['mean'] 
})

cluster_agg_sorted = cluster_agg.sort_values(('totalspent', 'sum'), ascending=False)

print(cluster_agg_sorted)


# In[130]:


from sklearn.metrics import silhouette_score

# Assume y_kmeans is the predicted cluster labels from the K-Means algorithm
silhouette_kmeans = silhouette_score(X, y_kmeans)
print('Silhouette Score for K-Means: ', silhouette_kmeans)


# If another cluster has high income but low spending, they might need a different approach to convert their potential into actual sales.

# # Hierarchical Clustering - Part 2

# In[134]:


from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)


# In[146]:


# Assume y_hc is the predicted cluster labels from the Agglomerative Clustering
silhouette_agglo = silhouette_score(X, y_hc)
print('Silhouette Score for Agglomerative Clustering: ', silhouette_agglo)


# In[136]:


import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.show()


# In[145]:


final_dataset['Cluster'] = y_hc 

cluster_agg = final_dataset.groupby('Cluster').agg({
    'totalspent': ['sum', 'count'],  
    'Income': ['mean'] 
})

cluster_agg_sorted = cluster_agg.sort_values(('totalspent', 'sum'), ascending=False)

print(cluster_agg_sorted)


# In[137]:


plt.figure(figsize=(15,8))
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 0')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 1')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 2')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers using Agglomerative Clustering')
plt.xlabel('Income')
plt.ylabel('Spending')
plt.legend()
plt.show()


# # DBSCAN

# In[152]:


# eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other.
# min_samples is the number of samples in a neighborhood for a point to be considered as a core point.
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.25, min_samples=4)
clusters = dbscan.fit_predict(X)
# Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
print(clusters)


# In[151]:


import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='plasma')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Clusters by DBSCAN')
plt.show()


# A large yellow cluster which represents the core area of high-density points found by DBSCAN. There are also a few scattered blue points which could represent noise or outlier points that are not part of the main cluster due to insufficient density in their local neighborhood.
