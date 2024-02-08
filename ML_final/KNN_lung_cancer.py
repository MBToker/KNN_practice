import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn import metrics
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset=pd.read_excel("survey_lung_cancer.xlsx")
dataset=dataset.dropna()
x=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,-1].values

#One hot encoding the 'GENDER' column
ct= ColumnTransformer(transformers=[("encoder",OneHotEncoder(),[0])],remainder="passthrough")
x=np.array(ct.fit_transform(x))
le=LabelEncoder()
y=le.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=1)
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

# Finding elbow value for KNN model
error_rate = {}
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate[i]=np.mean(pred_i != y_test)

optimal_neighbor_count=min(error_rate, key=error_rate.get)

# Building a KNN model
classifier=KNeighborsClassifier(n_neighbors=optimal_neighbor_count, metric="minkowski",p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)

# Finding results
cm=metrics.confusion_matrix(y_test,y_pred)
acs=accuracy_score(y_test,y_pred)
print("Accuracy: ",acs)

# Graph of elbow rule
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate.values(),color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# Graph of confusion matrix
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
cm_display.plot()
plt.show()




