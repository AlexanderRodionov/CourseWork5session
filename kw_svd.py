import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from numpy.linalg import svd
import random
from sys import exit as sys_exit 
from mpl_toolkits.mplot3d import Axes3D

num_obj = int(input("Количество объектов: "))
base_feat = int(input("Начальное количество признаков: "))
dec_feat = int(input("Количество признаков после уменьшения: "))

print("\n")

if dec_feat >= base_feat :
    print("Невозможно уменьшить размерность пространства! \n(Нельзя увеличивать пространство!)")
    sys_exit()
if base_feat <= 1 :
    print("Невозможно уменьшить размерность пространства! \n(Нельзя уменьшить пространство размерностью меньше 2!)")
    sys_exit()
if dec_feat < 1 :
    print("Невозможно уменьшить размерность пространства! \n(Нельзя уменьшить пространство до размерности меншье 1!)")
    sys_exit()
#random.seed(1)        
X=[]
for i in range(num_obj):
   X.append([random.randint(0,10) for _ in range(base_feat)])

X=np.array(X) #задаем матрицу явно
print("Матрица до преобразования SVD:")
print(X)

if base_feat == 3:
    xx=X[:,0]
    yy=X[:,1]
    zz=X[:,2]
    fig=plt.figure(1, figsize=(6, 5)) 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz)
    ax.set_title('ИСХОДНАЯ ИНФОРМАЦИЯ')
    ax.set_xlabel('первый признак',fontsize = 6)
    ax.set_ylabel('второй признак',fontsize = 6)
    ax.zaxis.set_label_text('третий признак',fontsize = 6)
    
elif base_feat == 2:
    xx=X[:,0]
    yy=X[:,1]
    fig=plt.figure(2)
    ax = fig.add_subplot(111)
    ax.scatter(xx, yy)
    ax.grid()
    ax.set_title('Исходная информация')
    ax.set_xlabel('первый признак')
    ax.set_ylabel('второй признак')

U, S, VT = svd(X)
print("Левые сингулярные векторы:") #столбец - это вектор
print(U)
print("Матрица сингулярных значений:")  #сингулярные числа - на диагонали
print(np.diag(S))
print("Правые сингулярные векторы:") #строка - это вектор
print(VT)
 
svd =  TruncatedSVD(n_components = dec_feat)
X_transf = svd.fit_transform(X) #выделяет наиболее информативные признаки
print('Сколько процентов объясняет каждый признак')
print(svd.explained_variance_ratio_)  
print('Сколько процентов объясняет уменьшенная матрица')
print(svd.explained_variance_ratio_.sum()) 

print("Преобразованная матрица после SVD:")
print(X_transf)

if  dec_feat == 3:
    xx=X_transf[:,0]
    yy=X_transf[:,1]
    zz=X_transf[:,2]
    fig=plt.figure(3, figsize=(6, 5)) 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz)
    ax.set_title('ИНФОРМАЦИЯ ПОСЛЕ SVD')
    ax.set_xlabel('первый признак',fontsize = 6)
    ax.set_ylabel('второй признак',fontsize = 6)
    ax.zaxis.set_label_text('третий признак',fontsize = 6)
    
elif dec_feat == 2:
    xx=X_transf[:,0]
    yy=X_transf[:,1]
    fig=plt.figure(4)
    ax = fig.add_subplot(111)
    ax.scatter(xx, yy)
    ax.grid()
    ax.set_title('ИНФОРМАЦИЯ ПОСЛЕ SVD')
    ax.set_xlabel('первый поризнак')
    ax.set_ylabel('второй признак')
    
    