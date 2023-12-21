#pandas,numply,matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Vectorlizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# model( SVM, DecisionTree) by sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

#model evaulate(confusion_matrix, accurarcy,recall,precision,f1, auc, roc)
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, auc, roc_curve

#勾配ブースィング
import xgboost as xgb


#データの前処理
#data: https://www.kaggle.com/datasets/sajid576/sql-injection-dataset/
data = pd.read_csv("../data/Modified_SQL_Dataset.csv")
print(data.info())

#text -> vector (vectorizer)
# vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b')
X = vectorizer.fit_transform(data["Query"])
Y = data["Label"]

#データの分割
X_train,X_test,y_train,y_test = train_test_split(X,Y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# ---- モデル構築 -----

#decision_tree
DecisionTree_model = DecisionTreeClassifier(random_state=42)
DecisionTree_model.fit(X_train,y_train)
Decision_tree_y_pred = DecisionTree_model.predict(X_test)

#SVC
SVC_model = SVC(random_state=42)
SVC_model.fit(X_train,y_train)
SVC_model_y_pred = SVC_model.predict(X_test)

#xgboost
xgb_model = xgb.XGBClassifier(objective="binary:logistic",random_state=42)
xgb_model.fit(X_train,y_train)
xgb_model_y_pred = xgb_model.predict(X_test)

result = {"Decision_tree":Decision_tree_y_pred,"SVC":SVC_model_y_pred,"XGB":xgb_model_y_pred}

#モデル評価 https://www.codexa.net/ml-evaluation-cls/

def model_evaluate(test,pred):
    accuracy = accuracy_score(test, pred)
    recall = recall_score(test,pred)
    precision = precision_score(test,pred)
    f1 = f1_score(test, pred)
    tn, fp, fn, tp = confusion_matrix(test, pred).ravel()
    return accuracy,recall,precision,f1,tn,fp,fn,tp


def plot_roc(model_name,test,pred):
    fpr,tpr,thresholds = roc_curve(test,pred)
    roc_auc = auc(fpr,tpr)
    fig = plt.figure(figsize=(5,5))
    plt.plot(fpr,tpr,label=model_name)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.grid()
    plt.savefig(f"../output/{model_name}.png")

for  model_name, pred in result.items():
    print(f"----------{model_name}-----------")
    accuracy,recall,precision,f1,tn,fp,fn,tp = model_evaluate(y_test,pred)
    print(f"正解率 : {accuracy}%")
    print(f"適合率 : {precision}%")
    print(f"再現率 : {recall}%")
    print(f"F1 : {f1}%")
    print(f"TN:{tn} FP:{fp} FN:{fn} TP:{tp}")
    print(confusion_matrix(y_test,pred))
    plot_roc(model_name,y_test,pred)