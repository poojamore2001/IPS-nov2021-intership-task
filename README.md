import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_context('notebook')
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("C:\\Users\\Dell\\Downloads\\iris.csv")
data.head()

print(data.shape[0])
data.columns.tolist()
data.dtypes
data['Species'] = data.Species.apply(lambda r: r.replace('Iris-', ''))
data.head()
data.Species.value_counts()
stats_df = data.describe()
stats_df.loc['range'] = stats_df.loc['max'] - stats_df.loc['min']

out_fields = ['mean','25%','50%','75%', 'range']
stats_df = stats_df.loc[out_fields]
stats_df.rename({'50%': 'median'}, inplace=True)
stats_df
data.groupby('Species').mean()
data.groupby('Species').median()
data.groupby('Species').agg(['mean', 'median'])  
data.groupby('Species').agg([np.mean, np.median]) 
agg_dict = {field: ['mean', 'median'] for field in data.columns if field != 'Species'}
agg_dict['PetalLengthCm'] = 'max'
pprint(agg_dict)
data.groupby('Species').agg(agg_dict)
ax = plt.axes()
ax.scatter(data.SepalLengthCm, data.SepalWidthCm)
ax.set(xlabel='Sepal Length (cm)',
       ylabel='Sepal Width (cm)',
       title='Sepal Length vs Width')
       
 ax = plt.axes()
ax.hist(data.PetalLengthCm, bins=25);
ax.set(xlabel='Petal Length (cm)', 
       ylabel='Frequency',
       title='Distribution of Petal Lengths')
 ax = data.iloc[:,1:].plot.hist(bins=25, alpha=0.5, figsize=(10,6))
ax.set_xlabel('Size (cm)')
axList = data.iloc[:,1:].hist(bins=25, figsize=(10,8))

# Add some x- and y- labels to first column and last row
for ax in axList.flatten():
    if ax.is_last_row():
        ax.set_xlabel('Size (cm)')
        
    if ax.is_first_col():
        ax.set_ylabel('Frequency')
data.iloc[:,1:].boxplot(by='Species', figsize=(10,8))
plot_data = (data.iloc[:,1:]
             .set_index('Species')
             .stack()
             .to_frame()
             .reset_index()
             .rename(columns={0:'size', 'level_1':'measurement'}))
plot_data.head()
sns.set_palette('muted')
plt.figure(figsize=(10,6))
sns.boxplot(x='measurement', y='size', 
            hue='Species', data=plot_data)
  sns.set_context('notebook')
sns.pairplot(data.iloc[:,1:], hue='Species')
corrmat = data.iloc[:,1:].corr()
sns.heatmap(corrmat, annot = True, square = True)
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
x = data.iloc[:,1:-1]
x.head()
y = pd.factorize(data['Species'])[0]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=21)
print("Train shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

def plot_confusion_matrix(y_test, y_preds, title):
    
    classes=['Setosa','Versicolor', 'Virginica']
    
    cm = confusion_matrix(y_test, y_preds)
    
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.1%}".format(value) for value in cm.flatten()/np.sum(cm)]
    
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(3,3)
    
    plt.style.use('seaborn')
    sns.set_context('notebook') 
    plt.figure(figsize=(8,6))
    ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
    ax.set_title(title)
    ax.set_xlabel('\nPredicted Species')
    ax.set_ylabel('Actual Species\n')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    plt.show()
    
def plot_roc(y_test, y_probs, title):
    fpr = {}
    tpr = {}
    roc_auc = {}
    thresh = {}
    classes = 3

    for i in range(classes):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test, y_probs[:,i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = ['salmon', 'teal', 'slateblue']
    species = ['Setosa' if i == 0 else 'Versicolor' if i == 1 else 'Virginica' for i in range(classes)]
    
    plt.figure(figsize=(9,6))
    for i, color in zip(range(classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2.5, alpha=0.8,
                 label='ROC Curve of {0} (AUC = {1:0.3f})'.format(species[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.title(title, fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive rate', fontsize=14)glm=LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(X_train, y_train)
glm_preds=glm.predict(X_test)
glm_probs=glm.predict_proba(X_test)
glm_acc=accuracy_score(y_test,glm_preds)
print("Test Set Accuracy: {:.1%}".format(glm_acc))
plot_confusion_matrix(y_test, y_preds=glm_preds, title='\nLogistic Regression Confusion Matrix')
plot_roc(y_test, y_probs=glm_probs, title='\nMulticlass ROC Curve of Iris Species\nLogistic Regression')
lda = LinearDiscriminantAnalysis().fit(X_train, y_train)
lda_preds=lda.predict(X_test)
lda_probs=lda.predict_proba(X_test)
lda_acc=accuracy_score(y_test,lda_preds)
print("Test Set Accuracy: {:.1%}".format(lda_acc))
plot_confusion_matrix(y_test, y_preds=lda_preds, title='\nLinear Discriminant Analysis Confusion Matrix')
plot_roc(y_test, y_probs=lda_probs, title='\nMulticlass ROC Curve of Iris Species\nLinear Discriminant Analysis')
svm = SVC(kernel='linear', C=1.2, probability=True).fit(X_train, y_train)
svm_preds=svm.predict(X_test)
svm_probs=svm.predict_proba(X_test)
svm_acc=accuracy_score(y_test,svm_preds)
print("Test Set Accuracy: {:.1%}".format(svm_acc))
plot_confusion_matrix(y_test, y_preds=svm_preds, title='\nLinear SVM Confusion Matrix')
plot_roc(y_test, y_probs=svm_probs, title='\nMulticlass ROC Curve of Iris Species\nLinear SVM')
data = {'Accuracy': [glm_acc, lda_acc, svm_acc]}
res = pd.DataFrame(data, index=['Logistic Regression', 'Linear Discriminant Analysis', 
                                'Linear SVM']).sort_values(by=['Accuracy'], ascending=False)
plt.figure(figsize=(9,6))
ax=sns.barplot(x=res.index, y='Accuracy', data=res, palette='Blues_d')
sns.despine(top=True, right=True, left=False, bottom=False)
plt.xlabel('Model', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0f}%'.format(x*100)))
for p in ax.patches:
    ax.annotate('{:.1f}%'.format(p.get_height()*100), (p.get_x()+0.4, p.get_height()), 
                ha='center', va='bottom', color= 'black')
plt.title('\nModel Accuracy on the Test Set', fontsize=20)
plt.show()
    plt.legend(loc='best', fontsize=14)
    plt.show()
