import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('tic_tac_toc.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())
col=data.columns.values
for i in col:
    print(data[i].value_counts())
    sn.countplot(data[i])
    plt.show()
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data['Top-left-square']=lab.fit_transform(data['top-left-square'])
data['Top-middle-square']=lab.fit_transform(data['top-middle-square'])
data['Top-right-square']=lab.fit_transform(data['top-right-square'])
data['Middle-left-square']=lab.fit_transform(data['middle-left-square'])
data['Middle-middle-square']=lab.fit_transform(data['middle-middle-square'])
data['Middle-right-square']=lab.fit_transform(data['middle-right-square'])
data['Bottom-left-square']=lab.fit_transform(data['bottom-left-square'])
data['Bottom-right-square']=lab.fit_transform(data['bottom-right-square'])
data['class']=lab.fit_transform(data['Class'])

from sklearn.model_selection import train_test_split
x=data[['Top-left-square','Top-right-square','Top-middle-square','Middle-left-square','Middle-middle-square',
        'Middle-right-square','Bottom-right-square','Bottom-left-square']]
colmns=x.columns.values
for i in colmns:
       for j in colmns:
              plt.plot(data[i].head(250),marker='o',label=f"{i}",color='red')
              plt.plot(data[j].head(250),marker="o",label=f'{j}',color='orange')
              plt.title(f'Its {i} vs {j}')
              plt.legend()
              plt.show()
import numpy as np
plt.figure(figsize=(17,6))
corr = x.corr(method='kendall')
my_m=np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()

y=data['class']
print(x['Middle-right-square'].value_counts())
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(max_iter=100)
lr.fit(x_train,y_train)
print(lr.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='gini')
tree.fit(x_train,y_train)
print(tree.score(x_test,y_test))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import VALID_METRICS
print(VALID_METRICS)
knn=KNeighborsClassifier(n_neighbors=4,weights='uniform',metric='manhattan')
knn.fit(x_train,y_train)
print(knn.score(x_test,y_test))
from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train,y_train)
print(xgb.score(x_test,y_test))