# import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from google.colab import files
uploaded = files.upload()
weather = pd.read_csv('seattle-weather.csv')
weather.shape
weather.head(35)
weather.isna().any()
weather.info()
sns.pairplot(data=weather, hue='weather')
sns.heatmap(data=weather.corr(),annot=True)
sns.countplot(data=weather, x='weather')
countrain=len(weather[weather.weather=="rain"])
countsun=len(weather[weather.weather=="sun"])
countdrizzle=len(weather[weather.weather=="drizzle"])
countsnow=len(weather[weather.weather=="snow"])
countfog=len(weather[weather.weather=="fog"])
print("Percent of Rain:{:2f}%".format((countrain/(len(weather.weather))*100)))
print("Percent of Sun:{:2f}%".format((countsun/(len(weather.weather))*100)))
print("Percent of Drizzle:{:2f}%".format((countdrizzle/(len(weather.weather))*100)))
print("Percent of Snow:{:2f}%".format((countsnow/(len(weather.weather))*100)))
print("Percent of Fog:{:2f}%".format((countfog/(len(weather.weather))*100)))
fig,axes = plt.subplots(2,2, figsize=(10,10))
cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
for i in range(4):
    sns.boxplot(x='weather', y=cols[i], data=weather, ax=axes[i%2,i//2])
fig.clear()
fig,axes = plt.subplots(2,2, figsize=(10,10))
for i in range(4):
    sns.histplot(kde=True, data=weather, x=cols[i], hue='weather', ax= axes[i%2, i//2])
# Remove unnecessary variables
df=weather.drop(["date"],axis=1)
# Skewed Distribution Treatment
df.precipitation=np.sqrt(df.precipitation)
df.wind=np.sqrt(df.wind)
def iqroutliers(data,x):
    Q1 = np.quantile(data[x],0.25)
    Q3 = np.quantile(data[x],0.75)
    iqr = Q3 - Q1
    outlier_values = data[x][(data[x]> iqr + 1.5 * iqr) | (data[x]< iqr - 1.5*iqr) ]
    outlier_index = outlier_values.index
    print(outlier_values)
    return list(outlier_values), list(outlier_index)
def removeoutliers(data,x):
    val,ind = iqroutliers(data,x)
    data.drop(ind, axis=0, inplace=True)
# Numerical variables
weather[["precipitation","temp_max","temp_min","wind"]].describe()
fig.clear()
fig,axes = plt.subplots(2,2, figsize=(10,10))
for i in range(4):
    sns.histplot(kde=True, data=weather, x=cols[i], hue='weather', ax= axes[i%2, i//2])
sns.set(style="darkgrid")
fig,axs=plt.subplots(2,2,figsize=(10,8))
sns.histplot(data=df,x="precipitation",kde=True,ax=axs[0,0],color='green')
sns.histplot(data=df,x="temp_max",kde=True,ax=axs[0,1],color='red')
sns.histplot(data=df,x="temp_min",kde=True,ax=axs[1,0],color='skyblue')
sns.histplot(data=df,x="wind",kde=True,ax=axs[1,1],color='orange')
plt.savefig('Histograms showing normalized distributions')
fig,axes = plt.subplots(2,2, figsize=(10,10))
cols = ['precipitation', 'temp_max', 'temp_min', 'wind']
for i in range(4):
    sns.boxplot(x='weather', y=cols[i], data=weather, ax=axes[i%2,i//2])
def normalizethis(data,cols):
    for x in cols:
        data[x] = data[x]/data[x].max()
normalizethis(weather,cols)
weather.head(35)
y= weather.pop('weather')
weather.pop('date')
#weather.pop('temp_min')
X= weather
#weather.pop('wind')
X=weather
# Label encoding
lc=LabelEncoder()
df["weather"]=lc.fit_transform(df["weather"])
import scipy
import re
import missingno as mso
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
X = df.loc[:, df.columns != "weather"].values.astype(int)
y = df["weather"].values
df.weather.unique()
df.head(35)
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=2)
xgbc = XGBClassifier()
gbc = GradientBoostingClassifier()
svc = SVC()
knn= KNeighborsClassifier()
xgbc.fit(x_train,y_train)
gbc.fit(x_train,y_train)
svc.fit(x_train, y_train)
knn.fit(x_train, y_train)y_pred_xgbc = xgbc.predict(x_test)
y_pred_gbc = gbc.predict(x_test)
y_pred_svc = svc.predict(x_test)
y_pred_knn = knn.predict(x_test)
knn_test_accuracy = knn.score(x_test, y_test)
gbc_test_accuracy = gbc.score(x_test, y_test)
svc_test_accuracy = svc.score(x_test, y_test)
xgbc_test_accuracy = xgbc.score(x_test, y_test)knn_train_accuracy = knn.score(x_train, y_train)
gbc_train_accuracy = gbc.score(x_train, y_train)
svc_train_accuracy = svc.score(x_train, y_train)
xgbc_train_accuracy = xgbc.score(x_train, y_train)
print('XGBC Train Accuracy = {:.2f}%'.format(xgbc_train_accuracy*100))
print('GBC Train Accuracy = {:.2f}%'.format(gbc_train_accuracy*100))
print('SVC Train Accuracy = {:.2f}%'.format(svc_train_accuracy*100))
print('KNN Train Accuracy = {:.2f}%'.format(knn_train_accuracy*100))
print('XGBC Test Accuracy = {:.2f}%'.format(xgbc_test_accuracy*100))
print('GBC Test Accuracy = {:.2f}%'.format(gbc_test_accuracy*100))
print('SVC Test Accuracy = {:.2f}%'.format(svc_test_accuracy*100))
print('KNN Test Accuracy = {:.2f}%'.format(knn_test_accuracy*100))
from sklearn.metrics import classification_report
print('XGBC\n',classification_report(y_test,y_pred_xgbc, zero_division=0))
print('SVC\n',classification_report(y_test,y_pred_svc, zero_division=0))
print('GBC\n',classification_report(y_test,y_pred_gbc, zero_division=0))
print('KNN\n',classification_report(y_test,y_pred_knn, zero_division=0))
# Comparision of training accuracies of the classifiers
acc = [knn_train_accuracy,svc_train_accuracy,gbc_train_accuracy, xgbc_train_accuracy]
classifiers = ['KNN', 'SVM', 'Gradient Boosting', 'XGBoost']
plt.bar(classifiers, acc)
ax = sns.barplot(x=classifiers, y=acc, palette = "mako", saturation =1.5)
plt.xlabel("Classification Models", fontsize = 20)
plt.ylabel("Accuracy", fontsize = 20)
plt.title("Comparision of training accuracies of the classifiers", fontsize = 20)
plt.savefig('Comparision of training accuracies of the classifiers')
plt.show()
# Comparision of testing accuracies of the classifiers
acc = [knn_test_accuracy,svc_test_accuracy,gbc_test_accuracy, xgbc_test_accuracy]
classifiers = ['KNN', 'SVM', 'Gradient Boosting', 'XGBoost']
plt.bar(classifiers, acc)
ax = sns.barplot(x=classifiers, y=acc, palette = "mako", saturation =1.5)
plt.xlabel("Classification Models", fontsize = 20)
plt.ylabel("Accuracy", fontsize = 20)
plt.title("Comparision of testing accuracies of the classifiers", fontsize = 20)
plt.savefig('Comparision of testing accuracies of the classifiers')
plt.show()
input=[[1.140175,8.9,2.8,2.469818]]
ot=gbc.predict(input)
print("The weather is:")
if(ot==0):
    print("Drizzle")
elif(ot==1):
    print("Fog")
elif(ot==2):
    print("Rain")
elif(ot==3):
    print("snow")
else:
    print("Sun")
