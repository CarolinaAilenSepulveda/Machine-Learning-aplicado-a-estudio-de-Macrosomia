#Librerias
import pandas as pd
pd.__version__
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import partial_dependence, plot_partial_dependence, permutation_importance
import itertools

#Dataframe revisado
macrosomia_dataframe = pd.read_csv("", sep=",", names=range(31))
#macrosomia_dataframe.head(363)
macrosomia_dataframe.columns=['PID','Historia clínica','Apellido','Nombre','Paridad','Peso materno','Altura materna','Fumadora','Historia de Diabetes Mellitus',
                            'Hipertensión crónica','Edad materna','Fecha de la exploración','Edad gestacional','+','FPP por ecografía','LCN','TN','Percentil TN','IP medio de las Arterias uterinas',
                           'equivalente a','Fecha(2)','EG(semanas)','Percentil IP arteria','EG(días)','Sexo del recien nacido','Peso del recien nacido','Score de Apgar1','Score de Apgar5','Apgar_10min',
                             'Tipo de parto','Indicación']
macrosomia_dataframe.head()

sns.pairplot(data=macrosomia_dataframe, hue='Sexo del recien nacido', height=10,vars=('Peso materno','Altura materna','Edad materna','Peso del recien nacido'))

#macrosomia_dataframe.columns
alt.Chart(macrosomia_dataframe).mark_point().encode(y='Peso del recien nacido',x='Peso materno',color='Sexo del recien nacido',tooltip=['Altura materna','Edad materna']).interactive()

alt.Chart(macrosomia_dataframe).mark_point().encode(y='Peso del recien nacido',x='Altura materna',color='Sexo del recien nacido',tooltip=['Peso materno','Edad materna']).interactive()

alt.Chart(macrosomia_dataframe).mark_point().encode(y='Peso del recien nacido',x='Edad materna',color='Sexo del recien nacido',tooltip=['Altura materna','Peso materno']).interactive()

nuevo_dataframe = pd.read_csv("", sep=",", names=range(4))
nuevo_dataframe.columns=['Semanas gestacion','Sexo del recien nacido','Peso recien nacido','Macrosomia']

#nuevo_dataframe.head()
alt.Chart(nuevo_dataframe).mark_point().encode(y='Macrosomia',x='Peso recien nacido',color='Sexo del recien nacido').interactive()

print ('Recien nacidos con macrosomia:43')
print ('Recien nacidos con peso normal:320')
print ('Como el total de recien nacidos es 363 y se tomo macrosomia para mas del 90%, se esperaria que 36 fueran macrosomicos')

macrosomia_dataframe_1 = pd.read_csv("", sep=",", names=range(2))
macrosomia_dataframe_1.head(363)
x_4 = macrosomia_dataframe_1[1].values.reshape(-1,1)
y_4= macrosomia_dataframe_1[0].values.reshape(-1,1)
modelo_4=LinearRegression().fit(x_4,y_4)

print ("----------------------------------------------MODELO REGRESION IMC----------------------------------------------")
print ("Termino independiente:",modelo_4.intercept_)
print("R cuadrado:",modelo_4.score(x_4,y_4))
y_predicta=modelo_4.predict(x_4,)
#print ("Peso recien nacido predicto:",y_predicta)

macrosomia_dataframe = pd.read_csv("", sep=",", names=range(31))
#macrosomia_dataframe.head(374)

#Modelo regresión Peso materno
x = macrosomia_dataframe[5].values.reshape(-1,1)
y= macrosomia_dataframe[25].values.reshape(-1,1)
modelo=LinearRegression().fit(x,y)

print ("----------------------------------------------MODELO REGRESION PESO MATERNO----------------------------------------------")
print ("Termino independiente:",modelo.intercept_)
print("R cuadrado:",modelo.score(x,y))
y_predicta=modelo.predict(x,)
#print ("Peso recien nacido predicto:",y_predicta)

#Prueba manual del modelo Peso materno
#x_prueba1=np.array([67,76,62]).reshape(-1,1)
#y_prueba1=modelo.predict(x_prueba1,)
#x_prueba2=np.array([80,90]).reshape(-1,1)
#y_prueba2=modelo.predict(x_prueba2)
#print ("Prueba1:",y_prueba1,",","Prueba2:",y_prueba2)

#Modelo regresión Altura materna
x_1 = macrosomia_dataframe[6].values.reshape(-1,1)
modelo_1=LinearRegression().fit(x_1,y)

print ("----------------------------------------------MODELO REGRESION ALTURA MATERNA--------------------------------------------")
print ("Termino independiente:",modelo_1.intercept_)
print("R cuadrado:",modelo_1.score(x_1,y))
y_predicta_1=modelo_1.predict(x_1,)
#print ("Peso recien nacido predicto:",y_predicta_1)

#Modelo regresión Edad materna
x_2 = macrosomia_dataframe[10].values.reshape(-1,1)
modelo_2=LinearRegression().fit(x_2,y)

print ("-----------------------------------------------MODELO REGRESION EDAD MATERNA----------------------------------------------")
print ("Termino independiente:",modelo_2.intercept_)
print("R cuadrado:",modelo_2.score(x_2,y))
y_predicta_2=modelo_2.predict(x_2,)
#print ("Peso recien nacido predicto:",y_predicta_2)

#Modelo multivariable
x_3 = macrosomia_dataframe[[5,6,10]]
modelo_3=LinearRegression().fit(x_3,y)

print ("-----------------------------------------------MODELO REGRESION MULTIVARIABLE----------------------------------------------")
print ("Termino independiente:",modelo_3.intercept_)
print("R cuadrado:",modelo_3.score(x_3,y))
y_predicta_3=modelo_3.predict(x_3,)
#print ("Peso recien nacido predicto:",y_predicta_3)


x = pd.read_csv("",sep=",", names=range(31))
x.columns=['PID','Historia clínica','Apellido','Nombre','Paridad','Peso materno','Altura materna','Fumadora','Historia de Diabetes Mellitus',
                            'Hipertensión crónica','Edad materna','Fecha de la exploración','Edad gestacional','+','FPP por ecografía','LCN','TN','Percentil TN','IP medio de las Arterias uterinas',
                           'equivalente a','Fecha(2)','EG(semanas)','Percentil IP arteria','EG(días)','Sexo del recien nacido','Peso del recien nacido','Score de Apgar1','Score de Apgar5','Apgar_10min',
                             'Tipo de parto','Indicación']

y=x.iloc[:,25].to_numpy()
x.drop(['PID','Historia clínica','Apellido','Nombre','Paridad','Fumadora','Historia de Diabetes Mellitus',
                        'Hipertensión crónica','Fecha de la exploración','Edad gestacional','+','FPP por ecografía','LCN','TN','IP medio de las Arterias uterinas',
                        'equivalente a','Fecha(2)','EG(semanas)','EG(días)','Score de Apgar1','Score de Apgar5','Apgar_10min',
                         'Tipo de parto','Indicación','Sexo del recien nacido','Peso del recien nacido','Percentil TN','Percentil IP arteria'],axis=1,inplace=True)
x_train,x_test,y_train,y_test=train_test_split (x,y,train_size=0.8,random_state=0)

#Normalización de datos
#normal=StandardScaler().fit(x_train)
#x_train= pd.DataFrame(normal.transform(x_train),columns=x.columns)
#x_test=pd.DataFrame(normal.transform(x_test),columns=x.columns)

modelolineal=LinearRegression().fit(x_train,y_train)

#Graficos de dependencia parcial 1 variable
#features=['Edad materna','Peso materno','Altura materna']
#figura_1, ax_1 = plt.subplots(figsize=(12, 10))
#grafico_1= plot_partial_dependence(modelolineal,x_test,features,
 #                                  ax=ax_1)

#Graficos de dependencia parcial 2 variables
interacciones=list(itertools.combinations(features,2))
figura_2, ax_2 = plt.subplots(figsize=(16, 10))
grafico_2= plot_partial_dependence(modelolineal,x_test,interacciones,
                                   grid_resolution=20,
                                   ax=ax_2)

macrosomia_dataframe = pd.read_csv("", sep=",", names=range(31))
macrosomia_dataframe.columns=['PID','Historia clínica','Apellido','Nombre','Paridad','Peso materno','Altura materna','Fumadora','Historia de Diabetes Mellitus',
                            'Hipertensión crónica','Edad materna','Fecha de la exploración','Edad gestacional','+','FPP por ecografía','LCN','TN','Percentil TN','IP medio de las Arterias uterinas',
                           'equivalente a','Fecha(2)','EG(semanas)','Percentil IP arteria','EG(días)','Sexo del recien nacido','Peso del recien nacido','Score de Apgar1','Score de Apgar5','Apgar_10min',
                             'Tipo de parto','Indicación']


alt.Chart(macrosomia_dataframe).mark_point().encode(y='Peso del recien nacido',x='TN').interactive()

alt.Chart(macrosomia_dataframe).mark_point().encode(y='TN',x='LCN').interactive()

alt.Chart(macrosomia_dataframe).mark_point().encode(y='Peso del recien nacido',x='LCN').interactive()

cuadro_tn_2 = pd.read_csv("", sep=",", names=range(4))

#Regresión percentil 5%
x = cuadro_tn_2[0].values.reshape(-1,1)
y= cuadro_tn_2[1].values.reshape(-1,1)

modelo_5=LinearRegression().fit(x,y)
print ("----------------------------------------------MODELO REGRESION 5%----------------------------------------------")
print ("Termino independiente:",modelo_5.intercept_)
print("R cuadrado:",modelo_5.score(x,y))

#Regresión percentil 50%
y_1= cuadro_tn_2[2].values.reshape(-1,1)

modelo_50=LinearRegression().fit(x,y_1)
print ("----------------------------------------------MODELO REGRESION 50%----------------------------------------------")
print ("Termino independiente:",modelo_50.intercept_)
print("R cuadrado:",modelo_50.score(x,y_1))

#Regresión percentil 95%
y_2= cuadro_tn_2[3].values.reshape(-1,1)

modelo_95=LinearRegression().fit(x,y_2)
print ("----------------------------------------------MODELO REGRESION 95%----------------------------------------------")
print ("Termino independiente:",modelo_95.intercept_)
print("R cuadrado:",modelo_95.score(x,y_2))

macrosomia_dataframe = pd.read_csv("", sep=",", names=range(29))
macrosomia_dataframe.head()

#Modelo regresión TN/LCN
x = macrosomia_dataframe[15].values.reshape(-1,1)
y= macrosomia_dataframe[16].values.reshape(-1,1)
modelo=LinearRegression().fit(x,y)

print ("----------------------------------------------MODELO REGRESION TN VS LCN----------------------------------------------")
print ("Termino independiente:",modelo.intercept_)
print("R cuadrado:",modelo.score(x,y))
y_predicta=modelo.predict(x,)

macrosomia_dataframe = pd.read_csv("", sep=",", names=range(31))
macrosomia_dataframe.columns=['PID','Historia clínica','Apellido','Nombre','Paridad','Peso materno','Altura materna','Fumadora','Historia de Diabetes Mellitus',
                            'Hipertensión crónica','Edad materna','Fecha de la exploración','Edad gestacional','+','FPP por ecografía','LCN','TN','Percentil TN','IP medio de las Arterias uterinas',
                           'equivalente a','Fecha(2)','EG(semanas)','Percentil IP arteria','EG(días)','Sexo del recien nacido','Peso del recien nacido','Score de Apgar1','Score de Apgar5','Apgar_10min',
                             'Tipo de parto','Indicación']

macrosomia_dataframe.head()
#alt.Chart(macrosomia_dataframe).mark_point().encode(x='Peso del recien nacido',y='Percentil TN',color='Sexo del recien nacido').interactive()

#macrosomia_dataframe.head()
alt.Chart(macrosomia_dataframe).mark_point().encode(x='Delta TN',y='Peso del recien nacido',color='Sexo del recien nacido',tooltip='Percentil TN').interactive()


alt.Chart(macrosomia_dataframe).mark_point().encode(x='IP medio de las Arterias uterinas',y='Peso del recien nacido',color='Sexo del recien nacido',tooltip='Percentil IP arteria').interactive()

macrosomia_dataframe = pd.read_csv("", sep=",", names=range(31))

x = macrosomia_dataframe[18].values.reshape(-1,1)
y= macrosomia_dataframe[25].values.reshape(-1,1)
modelo=LinearRegression().fit(x,y)

print ("----------------------------------------------MODELO REGRESION IP MEDIO----------------------------------------------")
print ("Termino independiente:",modelo.intercept_)
print("R cuadrado:",modelo.score(x,y))
#y_predicta=modelo.predict(x,)

macrosomia_dataframe = pd.read_csv("", sep=",", names=range(31))
macrosomia_dataframe.columns=['PID','Historia clínica','Apellido','Nombre','Paridad','Peso materno','Altura materna','Fumadora','Historia de Diabetes Mellitus',
                            'Hipertensión crónica','Edad materna','Fecha de la exploración','Edad gestacional','+','FPP por ecografía','LCN','TN','Percentil TN','IP medio de las Arterias uterinas',
                           'equivalente a','Fecha(2)','EG(semanas)','Percentil IP arteria','EG(días)','Sexo del recien nacido','Peso del recien nacido','Score de Apgar1','Score de Apgar5','Apgar_10min',
                             'Tipo de parto','Indicación']
macrosomia_dataframe.head()
alt.Chart(macrosomia_dataframe).mark_point().encode(x='Peso del recien nacido',y='Percentil IP arteria',color='Sexo del recien nacido').interactive()

alt.Chart(macrosomia_dataframe).mark_point().encode(x='Peso del recien nacido',y='IP medio de las Arterias uterinas',color='Percentil IP arteria').interactive()

alt.Chart(macrosomia_dataframe).mark_point().encode(x='Fecha de la exploración',y='IP medio de las Arterias uterinas',color='Percentil IP arteria').interactive()