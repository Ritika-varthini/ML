
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data =pd.read_csv('ex3.csv')
print("dataset preview")
print(data.head())
x=data[["age","income"]]
y=data["bought"]
print(x)

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LogisticRegression
regression=LogisticRegression()
regression.fit(xtr,ytr)
tepr=regression.predict(xte)
trpr=regression.predict(xtr)

from sklearn.metrics import accuracy_score,confusion_matrix, ConfusionMatrixDisplay
print("predictions:",tepr.tolist())
accuracy=accuracy_score(yte,tepr)
print("Accuracy:",accuracy)
cm=confusion_matrix(yte,tepr)
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Not Bought(0)","Bought(1)"])
disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix (Accuracy={accuracy:.2f})")
plt.show()
