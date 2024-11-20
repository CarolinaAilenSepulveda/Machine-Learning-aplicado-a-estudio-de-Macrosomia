import pandas  as pd
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,plot_roc_curve,classification_report
from sklearn.inspection import partial_dependence, plot_partial_dependence, permutation_importance
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree


data = pd.read_csv("", sep=",", names=range(4)) 
data.columns=['Semanas gestacion','Sexo del recien nacido','Peso recien nacido','Macrosomia']

print ('---------------------------------------------------------------')
print ('INFORMACIÓN CLASE DESBALANCEADA')
print ('---------------------------------------------------------------')
print ('Data shape:',data.shape)
print ('Numero de clases:',pd.value_counts(data['Macrosomia'],sort=True))
#pd.value_counts(data['Macrosomia'],sort=True).plot(kind='bar',rot=0)

y = data['Macrosomia']
x = data.drop(['Macrosomia','Sexo del recien nacido'],axis=1)
x_train,x_test,y_train,y_test=train_test_split (x,y,train_size=0.8,random_state=0)
features=['Semanas gestacion','Peso recien nacido']

#Tecnica SMOTE
print ('---------------------------------------------------------------')
print ('TÉCNICA SMOTE')
print ('---------------------------------------------------------------')
smote_1=SMOTE(ratio=0.5)
x_train_nueva_1,y_train_nueva_1=smote_1.fit_sample(x_train,y_train)
print ('Distribucion original:',Counter(y_train))
print ('Distribucion luego de aumentar la clase minoritaria con Smote:',Counter(y_train_nueva_1))

#Random forest
modeloforest_1=RandomForestClassifier(n_estimators=100).fit(x_train_nueva_1,y_train_nueva_1)
y_predicta_1=modeloforest_1.predict(x_test)
#print ('Valores de clase predicta:',y_predicta_1)
print ('---------------------------------------------------------------')
print ('RESULTADOS DEL CLASIFICADOR')
print ('---------------------------------------------------------------')
print ('Importancia de las variables:',modeloforest_1.feature_importances_)
print ('Matriz de confusión smote:',confusion_matrix(y_test,y_predicta_1))

print ('---------------------------------------------------------------')
print ('MATRIZ DE CONFUSIÓN Y DEPENDENCIAS PARCIALES')
print ('---------------------------------------------------------------')
#Grafico matriz confusion
mc=confusion_matrix(y_test,y_predicta_1)
ax3=plt.subplot()
sns.heatmap(mc,annot=True,ax=ax3)
ax3.set_xlabel('Prediccion');ax3.set_ylabel('Valor real')
ax3.set_title('Matriz de confusion')
ax3.xaxis.set_ticklabels(['0', '1']); ax3.yaxis.set_ticklabels(['0', '1'])

#Grafico dependecia parcial variables
fig1, ax1 = plt.subplots(figsize=(7, 5))
imagen_2= plot_partial_dependence(modeloforest_1, x_test, features,
                        ax=ax1) 

#Grafico curva ROC 
data = pd.read_csv("", sep=",", names=range(4)) 
data.columns=['Semanas gestacion','Sexo del recien nacido','Peso recien nacido','Macrosomia']

y = data['Macrosomia']
x = data.drop(['Macrosomia','Sexo del recien nacido'],axis=1)
x_train,x_test,y_train,y_test=train_test_split (x,y,train_size=0.8,random_state=0)
features=['Semanas gestacion','Peso recien nacido']

smote_1=SMOTE(ratio=0.5)
x_train_nueva_1,y_train_nueva_1=smote_1.fit_sample(x_train,y_train)
modeloforest_1=RandomForestClassifier(n_estimators=100).fit(x_train_nueva_1,y_train_nueva_1)
y_predicta_1=modeloforest_1.predict(x_test)

print ('------------------------------------------------------------------------')
print ('CURVA ROC (receiver operating characteristic)')
print ('------------------------------------------------------------------------')

#Grafico curva ROC
ax4=plt.gca()
fig4=plot_roc_curve(modeloforest_1,x_test,y_test,ax=ax4,alpha=0.8)
plt.show()

#Precision, recall y F1 score
print ('------------------------------------------------------------------------')
print ('PRECISION, RECALL (sensibilidad) y F1 SCORE')
print ('------------------------------------------------------------------------')
print (classification_report(y_test,y_predicta_1))

data = pd.read_csv("", sep=",", names=range(9)) 
data.columns=['peso materno','altura materna','edad materna','lcn','tn','ip medio','eg','peso recien nacido','clase']

print ('---------------------------------------------------------------')
print ('INFORMACIÓN CLASE DESBALANCEADA')
print ('---------------------------------------------------------------')
print ('Data shape:',data.shape)
print ('Numero de clases:',pd.value_counts(data['clase'],sort=True))
#pd.value_counts(data['clase'],sort=True).plot(kind='bar',rot=0)

y = data['clase']
x = data.drop(['clase','peso recien nacido'],axis=1)
x_train,x_test,y_train,y_test=train_test_split (x,y,train_size=0.8,random_state=0)
features=['peso materno','altura materna','edad materna','lcn','tn','ip medio','eg']


#Tecnica SMOTE
print ('---------------------------------------------------------------')
print ('TÉCNICA SMOTE')
print ('---------------------------------------------------------------')
smote_1=SMOTE(ratio=0.5)
x_train_nueva_1,y_train_nueva_1=smote_1.fit_sample(x_train,y_train)
print ('Distribucion original:',Counter(y_train))
print ('Distribucion luego de aumentar la clase minoritaria con Smote:',Counter(y_train_nueva_1))

#Random forest clasificador
modeloforest_1=RandomForestClassifier(n_estimators=100).fit(x_train_nueva_1,y_train_nueva_1)
y_predicta_1=modeloforest_1.predict(x_test)
#print ('Valores de clase predicta:',y_predicta_1)
print ('---------------------------------------------------------------')
print ('RESULTADOS DEL CLASIFICADOR')
print ('---------------------------------------------------------------')
print ('Importancia de las variables smote:',modeloforest_1.feature_importances_)
print ('Matriz de confusión smote:',confusion_matrix(y_test,y_predicta_1))

print ('---------------------------------------------------------------')
print ('MATRIZ DE CONFUSIÓN Y DEPENDENCIAS PARCIALES')
print ('---------------------------------------------------------------')

#Grafico matriz confusion
mc=confusion_matrix(y_test,y_predicta_1)
ax3=plt.subplot()
sns.heatmap(mc,annot=True,ax=ax3)
ax3.set_xlabel('Prediccion');ax3.set_ylabel('Valor real')
ax3.set_title('Matriz de confusion')
ax3.xaxis.set_ticklabels(['0', '1']); ax3.yaxis.set_ticklabels(['0', '1'])

#Grafico dependecia parcial de las variables
fig2, ax2 = plt.subplots(figsize=(10, 10))
imagen_2= plot_partial_dependence(modeloforest_1, x_test, features,
                        ax=ax2)
 
#Grafico curva ROC 
data = pd.read_csv("", sep=",", names=range(9)) 
data.columns=['peso materno','altura materna','edad materna','lcn','tn','ip medio','eg','peso recien nacido','clase']

y = data['clase']
x = data.drop(['clase','peso recien nacido'],axis=1)
x_train,x_test,y_train,y_test=train_test_split (x,y,train_size=0.8,random_state=0)
features=['peso materno','altura materna','edad materna','lcn','tn','ip medio','eg']

modelo_1=LinearRegression().fit(x_1,y)
y_predicta_1=modelo_1.predict(x,)
print ('------------------------------------------------------------------------')
print ('CURVA ROC (receiver operating characteristic)')
print ('------------------------------------------------------------------------')

#Grafico curva ROC
ax4=plt.gca()
fig4=plot_roc_curve(modeloforest_1,x_test,y_test,ax=ax4,alpha=0.8)
plt.show()

#Precision, recall y F1 score
print ('------------------------------------------------------------------------')
print ('PRECISION, RECALL (sensibilidad) y F1 SCORE')
print ('------------------------------------------------------------------------')
print (classification_report(y_test,y_predicta_1))


data = pd.read_csv("", sep=",", names=range(9)) 
data.columns=['peso materno','altura materna','edad materna','lcn','tn','ip medio','eg','peso recien nacido','clase']

y = data['clase']
x = data.drop(['clase','peso recien nacido','lcn'],axis=1)
x_train,x_test,y_train,y_test=train_test_split (x,y,train_size=0.8,random_state=0)
features=['peso materno','altura materna','edad materna','tn','ip medio','eg']

#Árbol de decisión
arbol = DecisionTreeRegressor(max_depth=3)
arbol.fit(x_train, y_train)
fig = plt.figure(figsize=(14,10))
_ = plot_tree(arbol, feature_names=features, filled=True)

print ('--------------------------------------------------------')
print('R2 arbol de decision: ' + str(arbol.score(x_test, y_test)))
print ('--------------------------------------------------------')
