# -*- coding: utf-8 -*-
"""
@author: Toktam Dehghani 
Email:dehghani.toktam@mail.um.ac.ir
Website: www.SmartDxCloud.ir
"""
import warnings
warnings.filterwarnings("ignore")
import math
import scipy
from scipy.stats import norm
import matplotlib.pyplot as pyplot
from sklearn import model_selection
import pandas as pd
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import plotting
from sklearn import svm
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn import svm
import time
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve 
from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
from sklearn.metrics import (brier_score_loss, precision_score, recall_score,f1_score,auc)
from sklearn.metrics import matthews_corrcoef 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import ConfusionMatrixDisplay
from collections import Counter
import researchpy as rp
import scipy.stats as stats
import shap

#----------------------------------------------------------------- Evaluation
def plot_Calibration():  
    print ('Calibration plots')
    print ('Calibration plots (reliability curve 2)', file=sourceFile)   
    fig=plt.figure(figsize=(15, 15))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))    
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for name ,clf in models:
        #clf=clf.fit(X_train, Y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(Y_test, prob_pos, n_bins=5)
       
        clf_score = brier_score_loss(Y_test, prob_pos, pos_label=1)
        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", lw=3,  label="%s (BS:=%1.3f)" % (name, clf_score))  
        ax2.hist(prob_pos, range=(0, 1), bins=5, label=name,  histtype="step", lw=3)
    
    ax1.set_ylabel("Predicted Probability",fontsize= 28)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="center left", ncol=1,fontsize= 24, bbox_to_anchor=(1.05, 0.5))
    #ax1.set_title('Calibration plots (reliability curve)',fontsize= 30)
    
    ax2.set_xlabel("True Probability of Passing Exam",fontsize= 28)
    ax2.set_ylabel("Count",fontsize= 28)
    ax2.legend(loc="center left", ncol=1,fontsize= 24, bbox_to_anchor=(1.05, 0.5))
    
    plt.tight_layout()
    plt.show()
    fig.savefig('Calibration plots (reliability curve)2'+'.svg', format='svg', dpi=1200)

def plot_roc():        
    print ('Receiver operating characteristic (ROC)')
    print ('Receiver operating characteristic (ROC)', file=sourceFile)
    #sns.set(style="white",font='sans-serif',font_scale=1.5)
    fig, axs = plt.subplots(figsize=(10,10))
    for name, model in models:
        #model=model.fit(X_train,Y_train)
        y_predict=model.predict(X_test)
        y_predict=y_predict.ravel()
        LR_roc_auc=roc_auc_score(Y_test,y_predict.round(),average='macro')
        Y_test_predict=model.predict_proba(X_test)
        fpr,tpr,thresholds=roc_curve(Y_test,Y_test_predict[:,1]) 
        axs.plot(fpr,tpr,label=name+' (AUC ='+str(round(LR_roc_auc,2))+")")
    axs.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    axs.set_xlabel("1 - specificity",fontsize= 30)
    axs.set_ylabel("Sensitivity",fontsize= 30)
    axs.set_ylim([0.0,1.05])
    axs.set_xlim([0.0,1.05])
    axs.legend(loc="center left", ncol=1,fontsize= 24, bbox_to_anchor=(1.05, 0.5))
    fig.savefig('prediction_ROC_AUC'+'.svg', format='svg', dpi=1200)
    
def plot_Precision_Recall():
    print ('Precision-Recall curve')
    print ('Precision-Recall curve', file=sourceFile)     
    fig1, axs1 = plt.subplots(figsize=(10,10))
    for name, model in models:      
          #model.fit(X_train, Y_train)
          y_score = model.predict(X_test)
          # y_score=y_score.ravel()
          Y_test_predict=model.predict_proba(X_test)
          precision, recall, _ = precision_recall_curve(Y_test, Y_test_predict[:, 1].ravel()) 
          lr_f1 =f1_score(Y_test, y_score.round(),average='weighted')
          axs1.plot(recall, precision,label="%s (F1 = %.2f)" % (name, round(lr_f1,2) ))
    axs1.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    axs1.set_ylim([0.0,1.05])
    axs1.set_xlim([0.0,1.05])
    axs1.set_ylabel('Precision',fontsize= 30)
    axs1.set_xlabel('Recall',fontsize= 30) 
    axs1.legend(loc="center left", ncol=1,fontsize= 24, bbox_to_anchor=(1.05, 0.5))
    fig1.savefig('Precision-Recall curve'+'.svg', format='svg', dpi=1200)
    
def save_output():  
    print ('save results of models')
    print ('save results of models', file=sourceFile)    
    Y_test.to_csv(r'Y_test.csv')
    for name, model in models: 
          Y_test_predict=np.round(abs(model.predict(X_test)))
          Y_test_predict.tofile('Y-test_predict_'+name+'.csv',sep=',',format="%s")  

def avaluate(model):
    print("\n***********************\n Classification_report")
    print("\n***********************\n Classification_report", file=sourceFile) 
    print("\nTraining", file=sourceFile)
    y_score = model.predict(X_train).round()
    y_score=y_score.ravel()
    Y_train_predict=model.predict_proba(X_train)
    print("roc_auc:: %1.3f\n" % roc_auc_score(Y_train, y_score,average='macro'), file=sourceFile)    
    print("Average Precision: %1.3f" % average_precision_score(Y_train, y_score,average='weighted'), file=sourceFile)
    print("Precision: %1.3f" % precision_score(Y_train, y_score,average='weighted'), file=sourceFile)
    print("Recall: %1.3f" % recall_score(Y_train, y_score,average='weighted'), file=sourceFile) 
    print("tn, fp, fn, tp \n" , file=sourceFile)
    print(confusion_matrix (Y_train, y_score), file=sourceFile)
    print("accuracy:: %1.3f" % accuracy_score(Y_train, y_score), file=sourceFile)
    print("F1_weighted: %1.3f" % f1_score(Y_train, y_score,average='weighted'), file=sourceFile) 
    print("MCC:: %1.3f" % matthews_corrcoef(Y_train, y_score), file=sourceFile) 
    print("Brier scores: %1.3f (smaller is better)" % brier_score_loss(Y_train, Y_train_predict[:, 1],pos_label=1), file=sourceFile)         
    print(classification_report(Y_train, y_score), file=sourceFile) 
    print("\nTest", file=sourceFile)          
    y_score = model.predict(X_test).round()
    y_score=y_score.ravel()
    Y_test_predict=model.predict_proba(X_test)
    print("roc_auc:: %1.3f" % roc_auc_score(Y_test, y_score,average='macro'), file=sourceFile)           
    print("Average Precision (area PR): %1.3f" % average_precision_score(Y_test, y_score,average='weighted'), file=sourceFile) 
    print("Precision: %1.3f" % precision_score(Y_test, y_score,average='weighted'), file=sourceFile) 
    print("Recall: %1.3f" % recall_score(Y_test, y_score,average='weighted'), file=sourceFile) 
    print("tn, fp, fn, tp" , file=sourceFile)
    print(confusion_matrix (Y_test, y_score), file=sourceFile)  
    print("accuracy:: %1.3f" % accuracy_score(Y_test, y_score), file=sourceFile)
    print("F1: %1.3f" % f1_score(Y_test, y_score ,average='weighted'), file=sourceFile) 
    print("MCC:: %1.3f" % matthews_corrcoef(Y_test, y_score), file=sourceFile)
    print("Brier scores: %1.3f (smaller is better)" % brier_score_loss(Y_test, Y_test_predict[:, 1],pos_label=1), file=sourceFile)         
    print('Mean squared error: %.2f'% mean_squared_error(Y_test, y_score, squared=False), file=sourceFile) 
    print('Coefficient of determination: %.2f' % r2_score(Y_test, y_score), file=sourceFile) 
    print(classification_report(Y_test, y_score), file=sourceFile) 
        
def plot_confusion_matrix():
    print ('plot_confusion_matrix')
    print ('plot_confusion_matrix', file=sourceFile)  
    for name, model in models:      
        y_score = model.predict(X_test)
        y_score=y_score.ravel()
        fig8 = plt.figure(figsize=(15,15))
        cm=confusion_matrix(Y_test, y_score,normalize='all')        
        plt.plot=sns.heatmap(cm.round(2), annot=True, linewidths=.5, fmt='g',cmap="Spectral_r",cbar_kws={"orientation": "horizontal"} )
        plt.suptitle('Confusion_matrix ('+ str(name) +')',fontsize= 30) 
        plt.xlabel('Predicted class',fontsize= 28)
        plt.ylabel('Actual class',fontsize= 28) 
        fig8.savefig('confusion_matrix_'+str(name)+'.svg', format='svg', dpi=1200)
    
def plot_comparing_accurecy():
    print ('Models comparision (accuracy)')
    print ('Models comparision (accuracy)', file=sourceFile) 
    results=[]
    names=[]
    fig4 = plt.figure(figsize=(40,20))
    for name, model in models:  
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=12)
	    result=model_selection.cross_val_score(model,X_test, Y_test,cv=CV,scoring='accuracy',n_jobs=-1)
        results.append(result)
        names.append(name)
    fig4=plt.figure()
    fig4.suptitle('Models Accuracy Comparision',fontsize= 20)
    ax=fig4.add_subplot(111)
    plt.violinplot(results)
    ax.set_xticklabels(names,fontsize= 12)
    ax.set_yticklabels([],fontsize= 12)
    plt.xlabel('Methods',fontsize= 14)
    plt.ylabel('Accuracy',fontsize= 14)
    fig4.savefig('Comparision_accurecy'+'.svg', format='svg', dpi=1200)

def plot_comparing_roc_auc():
    print ('Models comparision (roc_auc)')
    print ('Models comparision (roc_auc)', file=sourceFile) 
    results=[]
    names=[]
    fig44 = plt.figure(figsize=(20,20))    
    for name, model in models: 
        result=model_selection.cross_val_score(model,X_test, Y_test,cv=CV,scoring='roc_auc')
        results.append(result)
        names.append(name)
    fig44=plt.figure()
    fig44.suptitle('Models comparision (roc_auc)',fontsize= 28)
    ax=fig44.add_subplot(111)
    plt.boxplot(results,1, '')
    ax.set_xticklabels(names,fontsize= 28)
    ax.set_yticklabels(results,fontsize= 28)
    plt.xticks(rotation=28)
    fig44.savefig('Comparision_roc_auc'+'.svg', format='svg', dpi=1200)
    

#_________________________________________________________________________________________________________________________________    
#_________________________________________________________________________________________________________________________________       
#__________________________________________________  Load Data ________________________________________________________________________
#_________________________________________________________________________________________________________________________________ 
print("1.Load data")
report_file='log_file_2022.txt'   
input_file="Sample.xlsx"
input_sheet='Sheet1'
sourceFile = open(report_file, 'w')
stu_info_payeh= pd.read_excel(input_file, sheet_name=input_sheet)
print(stu_info_payeh.info())
#_________________________________________________________________________________________________________________________________    
#_________________________________________________________________________________________________________________________________       
print("2. Dealing with missing values ")
dt=stu_info_payeh[['Anatomical Sciences ','Physiology','Biochemistry','Technical English Language','Bacteriology','Virology','Parasitology and Entomology','Mycology','Principles of Epidemiology','Public Health','General Knowledge','GPA','Age at Entrance ', 'Age at CMBSE','Gender', 'Residency Status', 'Entrance Semester', 'Type of Admission','Status in the CMBSE','Normalized CMBSE Score']]#,'No of Attemp']]
dt=dt.dropna(subset=['Status in the CMBSE'])
#---------------------------------------------------------- Normalization
print("3. Normalization ")
names=['Gender', 'Residency Status', 'Entrance Semester', 'Type of Admission']
for name in names: 
    df_one = pd.get_dummies(data=dt, columns=[str(name)])
    df_two = pd.concat((df_one, dt[str(name)]), axis=1)
    df_two = df_two.drop([str(name)], axis=1)
    dt=df_two
dt.rename(columns = {'Residency Status_Local':'Residency Status'}, inplace = True)
dt.rename(columns = {'Entrance Semester_First':'Entrance Semester'}, inplace = True)
dt.rename(columns = {'Gender_Male':'Gender'}, inplace = True)
dt.rename(columns = {'Type of Admission_Daily Course':'Type of Admission'}, inplace = True)
dt[['Status in the CMBSE'] ]=dt[['Status in the CMBSE'] ].replace("Fail", 0)
dt[['Status in the CMBSE'] ]=dt[['Status in the CMBSE'] ].replace("Pass", 1)
#print(dt[['Status in the CMBSE']])
#----------------------------------------------------------- Apply Weights of courses
print("4. Apply Weights of courses ")
dt= dt[['Anatomical Sciences ','Physiology','Biochemistry','Technical English Language','Bacteriology','Virology','Parasitology and Entomology','Mycology','Principles of Epidemiology','Public Health','General Knowledge','GPA','Gender', 'Residency Status', 'Entrance Semester', 'Type of Admission','Status in the CMBSE']]
columns = ['Physiology', 'Biochemistry', 'Bacteriology',
'Parasitology and Entomology', 'Mycology', 'Virology',
'Anatomical Sciences ', 'Public Health',
'Principles of Epidemiology', 'Technical English Language',
'General Knowledge']
weights=[0.27,0.18,0.1,0.1,0.08,0.025,0.06,0.025,0.06,0.05,0.05]
for i in range (0, len(weights)): 
    dt[columns[i]]=dt[columns[i]]*weights[i]
#----------------------------------------------------------  Transformation 
print("5. Transformation") 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
ss_train =  MinMaxScaler()#
dt[['Physiology', 'Biochemistry', 'Bacteriology',
'Parasitology and Entomology', 'Mycology', 'Virology',
'Anatomical Sciences ', 'Public Health',
'Principles of Epidemiology', 'Technical English Language',
'General Knowledge','GPA']]= ss_train.fit_transform(dt[['Physiology', 'Biochemistry', 'Bacteriology',
'Parasitology and Entomology', 'Mycology', 'Virology',
'Anatomical Sciences ', 'Public Health',
'Principles of Epidemiology', 'Technical English Language',
'General Knowledge','GPA']])
                                                  
for name in columns:        
    dt[str(name)] = pd.qcut(dt[str(name)] ,20, labels=False, duplicates='drop')  
    
dt[['Physiology', 'Biochemistry', 'Bacteriology',
'Parasitology and Entomology', 'Mycology', 'Virology',
'Anatomical Sciences ', 'Public Health',
'Principles of Epidemiology', 'Technical English Language',
'General Knowledge','GPA']]= ss_train.fit_transform(dt[['Physiology', 'Biochemistry', 'Bacteriology',
'Parasitology and Entomology', 'Mycology', 'Virology',
'Anatomical Sciences ', 'Public Health',
'Principles of Epidemiology', 'Technical English Language',
'General Knowledge','GPA']])    
dt[['Physiology', 'Biochemistry', 'Bacteriology',
'Parasitology and Entomology', 'Mycology', 'Virology',
'Anatomical Sciences ', 'Public Health',
'Principles of Epidemiology', 'Technical English Language',
'General Knowledge','GPA']]= dt[['Physiology', 'Biochemistry', 'Bacteriology',
'Parasitology and Entomology', 'Mycology', 'Virology',
'Anatomical Sciences ', 'Public Health',
'Principles of Epidemiology', 'Technical English Language',
'General Knowledge','GPA']].round(3)  
#--------------------------------------------------Generate test and train   
print("6. Generate test and train ")                            
dt_train, dt_test = model_selection.train_test_split(dt, test_size=0.33, random_state=42, shuffle=True, stratify=dt["Status in the CMBSE"])
X_train = dt_train[['Anatomical Sciences ','Physiology','Biochemistry','Technical English Language','Bacteriology','Virology','Parasitology and Entomology','Mycology','Principles of Epidemiology','Public Health','General Knowledge','GPA','Gender', 'Residency Status', 'Entrance Semester', 'Type of Admission']]#,'No of Attemp']]
X_test= dt_test[['Anatomical Sciences ','Physiology','Biochemistry','Technical English Language','Bacteriology','Virology','Parasitology and Entomology','Mycology','Principles of Epidemiology','Public Health','General Knowledge','GPA','Gender', 'Residency Status', 'Entrance Semester', 'Type of Admission']]#,'No of Attemp']]
Y_train=dt_train[['Status in the CMBSE']]
Y_test=dt_test[['Status in the CMBSE']]
X_train2 = pd.DataFrame(X_train)
X_train2.to_csv(r'X_train.csv')
Y_train2 = pd.DataFrame(Y_train)
Y_train2.to_csv(r'Y_train.csv')
X_test2 = pd.DataFrame(X_test)
X_test2.to_csv(r'X_test.csv')
Y_test2 = pd.DataFrame(Y_test)
Y_test2.to_csv(r'Y_test.csv')
#--------------------------------------------------Resampling    
print("7. ReSmoting") 
from imblearn.combine import  SMOTETomek 
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
sm= SMOTETomek(random_state=1)
X_train,Y_train = sm.fit_resample(X_train,Y_train)
X_train2 = pd.DataFrame(X_train)
X_train2.to_csv(r'X_train_transform.csv')
Y_train2 = pd.DataFrame(Y_train)
Y_train2.to_csv(r'Y_train_transform.csv')
#-------------------------------------------------Modeling
print("8. Modeling")    
models2=[]
models=[]
results=[]
names=[]
Refit='roc_auc'
Scoring = [ 'roc_auc','precision','recall','accuracy','f1']
#-------------------------------------------------K-fold cross validation
print("9. K-fold cross validation") 
from sklearn.model_selection import RepeatedStratifiedKFold
CV=RepeatedStratifiedKFold(n_splits=5,random_state=12,n_repeats=1)
print("10. LR") 
print("\n LR",file=sourceFile)
random_grid={ }
clf = GridSearchCV(LogisticRegression(), random_grid, refit=Refit,    scoring = Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
t0 = time.time()
lr=clf.fit(X_train, Y_train)
kr_fit = time.time() - t0
print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
models.append(("LR",lr.best_estimator_))
models2.append(("LR",lr.best_estimator_))
print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
avaluate(lr.best_estimator_)

# from sklearn.tree import DecisionTreeClassifier
# print("DT")
# print("\n DT", file=sourceFile)
# random_grid={}
# clf = GridSearchCV( DecisionTreeClassifier(), random_grid, refit=Refit,    scoring = Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
# t0 = time.time()
# lr=clf.fit(X_train, Y_train)
# kr_fit = time.time() - t0
# print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
# models.append(("DT",lr.best_estimator_))
# models2.append(("DT",lr.best_estimator_))
# print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
# print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
# print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
# avaluate(lr.best_estimator_)

# from sklearn.naive_bayes import GaussianNB
# print("Naive Bayes")
# print("\n Naive Bayes", file=sourceFile)
# random_grid={}
# clf = GridSearchCV( DecisionTreeClassifier(), random_grid, refit=Refit,    scoring = Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
# t0 = time.time()
# lr=clf.fit(X_train, Y_train)
# kr_fit = time.time() - t0
# print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
# models.append(("Naive Bayes",lr.best_estimator_))
# models2.append(("Naive Bayes",lr.best_estimator_))
# print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
# print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
# print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
# avaluate(lr.best_estimator_)


print("11. SVM")
print("\n SVM", file=sourceFile)
random_grid={}
clf = GridSearchCV( svm.SVC(probability=True), random_grid, refit=Refit,    scoring = Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
t0 = time.time()
lr=clf.fit(X_train, Y_train)
kr_fit = time.time() - t0
print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
models.append(("SVM",lr.best_estimator_))
models2.append(("SVM",lr.best_estimator_))
print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
avaluate(lr.best_estimator_)

print("12. KNN")
print("\n KNN",file=sourceFile)
random_grid={
    #'n_neighbors': [ 5,10],
         #'weights': ['distance'],
         #'leaf_size': [15]
         }
clf = GridSearchCV( KNeighborsClassifier(), random_grid, refit=Refit,    scoring = Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
t0 = time.time()
lr=clf.fit(X_train, Y_train)
kr_fit = time.time() - t0
print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
models.append(("KNN",lr.best_estimator_))
models2.append(("KNN",lr.best_estimator_))
print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
avaluate(lr.best_estimator_)

# print("HardVoting")
# print("\n HardVoting",file=sourceFile)
# from sklearn.ensemble import VotingClassifier
# final_model = VotingClassifier(    estimators=[svm.SVC(probability=True),LogisticRegression(),KNeighborsClassifier()], voting='hard',n_jobs=-1) 
# # training all the model on the train dataset
# lr=lr.fit(X_train, Y_train)
# models.append(("HardVoting",lr.best_estimator_))
# models2.append(("HardVoting",lr.best_estimator_))
# print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
# print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
# print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
# avaluate(lr.best_estimator_)

print("13.  SoftVoting")
print("\n Voting",file=sourceFile)
from sklearn.ensemble import VotingClassifier
final_model = VotingClassifier(    estimators=[svm.SVC(probability=True),LogisticRegression(),KNeighborsClassifier()], voting='hard',n_jobs=-1) 
# training all the model on the train dataset
lr=lr.fit(X_train, Y_train)
models.append(("Voting",lr.best_estimator_))
models2.append(("Voting",lr.best_estimator_))
print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
avaluate(lr.best_estimator_)

print("14.  RF")
print("\n RF",file=sourceFile)
#from sklearn.ensemble import RandomForestRegressor
random_grid={
    'n_estimators': [ 10000],
    #'max_depth' : [5,10,15,20],
    'criterion' :[ 'entropy'],
    #'class_weight' : [ 'balanced', 'None'],
    #'max_depth': [10, 20, None],
    # 'max_features': ['auto', 'sqrt'],
    # 'min_samples_leaf': [1, 2, 4],
    # 'min_samples_split': [2, 5, 10],
    #'ccp_alpha': [0.1],
    'bootstrap' : [True],
    'oob_score': [False]
    }                        
clf = GridSearchCV(RandomForestClassifier( n_jobs=-1, random_state=1), random_grid,   refit=Refit,  scoring =Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
t0 = time.time()
lr=clf.fit(X_train, Y_train)
kr_fit = time.time() - t0
print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
models.append(("RF",lr.best_estimator_))
models2.append(("RF",lr.best_estimator_))
print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
avaluate(lr.best_estimator_)

print("15. Bagging")
print("\n BAgging",file=sourceFile)
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
random_grid={ 
  #"n_estimators" : [5000],
  #'base_estimator' : [ LogisticRegression()],
#max_depth = [5, 10, 15, 25, 30]
#"max_samples" : [5],
#"max_features" : [ 15]
      }
clf = GridSearchCV(BaggingClassifier(), random_grid,  refit=Refit,   scoring =Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
t0 = time.time()
lr=clf.fit(X_train, Y_train)
kr_fit = time.time() - t0
print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
models.append(("BG",lr.best_estimator_))
models2.append(("BG",lr.best_estimator_))
print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
avaluate(lr.best_estimator_)

print("16. ADA")
print("\n ADA",file=sourceFile)
random_grid={
     #'n_estimators' : [1000],
     #'base_estimator' : [ LogisticRegression()],
     #'learning_rate': [0.01],
     #'random_state' : [42 ] ,
     #'algorithm': [ 'SAMME.R']
     }
clf = GridSearchCV(AdaBoostClassifier(), random_grid,  refit=Refit,    scoring =Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
t0 = time.time()
lr=clf.fit(X_train, Y_train)
kr_fit = time.time() - t0
print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
models.append(("ADA",lr.best_estimator_))
models2.append(("ADA",lr.best_estimator_))
print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
avaluate(lr.best_estimator_)

print("17.  XGB")
print("\n XGB",file=sourceFile)
from sklearn.ensemble import HistGradientBoostingClassifier
random_grid={
          #'max_depth': [ 3, 18, 1],
        #'gamma': [ 1,10],
        #'reg_alpha' : [40,180,1],
        #'reg_lambda' : [ 0,1],
        #'colsample_bytree' : [ 0.5,1],
        #'min_child_weight' : [ 0, 10, 1],
        #'n_estimators': [10000],
        #'seed':[ 0],
        #'objective':['binary:logistic']
        'max_leaf_nodes' : [10, 20, 30,]
}
#clf = GridSearchCV(GradientBoostingClassifier(learning_rate=0.01,max_features='sqrt',subsample=0.8,random_state=10,n_estimators=1000), random_grid,  refit=Refit,    scoring =Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
clf = GridSearchCV(HistGradientBoostingClassifier(learning_rate=0.01,random_state=42), random_grid,  refit=Refit,    scoring =Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
t0 = time.time()
lr=clf.fit(X_train, Y_train)
kr_fit = time.time() - t0
print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
models.append(("XGB",lr.best_estimator_))
models2.append(("XGB",lr.best_estimator_))
print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
avaluate(lr.best_estimator_)


# from xgboost import XGBClassifier
# print("XGB")
# print("\n XGB",file=sourceFile)
# random_grid={
#          #'max_depth': [ 3, 18, 1],
#         #'gamma': [ 1,10],
#         #'reg_alpha' : [40,180,1],
#         #'reg_lambda' : [ 0,1],
#         #'colsample_bytree' : [ 0.5,1],
#         #'min_child_weight' : [ 0, 10, 1],
#         #'n_estimators': [10000],
#         #'seed':[ 0],
#         #'objective':['binary:logistic']
# }
# clf = GridSearchCV(GradientBoostingClassifier(learning_rate=0.01,max_features='sqrt',random_state=12,n_estimators=5000), random_grid,  refit=Refit,    scoring =Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
# #clf = GridSearchCV(XGBClassifier(), random_grid,  refit=Refit,    scoring =Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
# t0 = time.time()
# lr=clf.fit(X_train, Y_train)
# kr_fit = time.time() - t0
# print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
# models.append(("XGB",lr.best_estimator_))
# models2.append(("XGB",lr.best_estimator_))
# print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
# print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
# print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
# avaluate(lr.best_estimator_)

from xgboost import XGBClassifier
print("18.  stack")
print("\n stack",file=sourceFile)
random_grid={}        
rfc = RandomForestClassifier( n_jobs=-1,criterion='entropy', random_state=1,n_estimators=10000,ccp_alpha=0.1,    bootstrap=True,    oob_score=False)
xgb = HistGradientBoostingClassifier(learning_rate=0.01,random_state=1)
lr = LogisticRegression()
ada=AdaBoostClassifier()
clf2 = [('rfc',rfc),('xgb',xgb),('ADA',ada)] #list of (str, estimator)
from sklearn.ensemble import StackingClassifier
stack_model = StackingClassifier( estimators = clf2,final_estimator = xgb)
clf = GridSearchCV(stack_model, random_grid,  refit=Refit,    scoring =Scoring,  cv = CV,     verbose=True,  n_jobs=-1)
t0 = time.time()
lr=clf.fit(X_train, Y_train)
kr_fit = time.time() - t0
print("complexity and bandwidth selected and model fitted in %.3f s"      % kr_fit , file=sourceFile)
models.append(("stack",lr.best_estimator_))
models2.append(("stack",lr.best_estimator_))
print("score on test: " + str(lr.score(X_test, Y_test)), file=sourceFile)
print("score on train: "+ str(lr.score(X_train, Y_train)), file=sourceFile)
print('parametrs:'+ str(lr.best_estimator_.get_params()), file=sourceFile)
avaluate(lr.best_estimator_)
#-------------------------------------------------------------- Evaluation
print("19.  Evaluate", file=sourceFile)
save_output()
plot_confusion_matrix()
plot_Calibration()
plot_Precision_Recall()
plot_comparing_accurecy()
plot_roc()
sourceFile.close()     


