#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#reading data
ad_data = pd.read_csv('advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()

#exploratory data analysis
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.show()
sns.jointplot('Age', 'Area Income', data=ad_data)
plt.show()
sns.jointplot('Age','Daily Time Spent on Site',kind='kde',data=ad_data,color='red')
plt.show()
sns.jointplot('Daily Time Spent on Site','Daily Internet Usage',data=ad_data,color='green')
plt.show()
sns.pairplot(ad_data,hue='Clicked on Ad',kind='scatter',diag_kind='hist')
plt.show()

#logistic regression
X = ad_data[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = ad_data['Clicked on Ad']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

#evaluating
pred = logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))