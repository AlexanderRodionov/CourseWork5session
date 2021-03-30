import numpy as np
import matplotlib.pyplot as plt
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
   
print('Матрица до преобразования PCA:')
print(X)
m = np.mean(X, axis = 0)
print("Смещение по координатам для центрирования: ")
print(m)

Xcentered=[]
for t in range(num_obj):
    Xline=[] #центруем каждый объект
    Xline = np.subtract(X[t], m)
    Xcentered.append(Xline)

Xcentered=np.array(Xcentered)

print("Отцентрованная матрица")
print(Xcentered) 

if base_feat == 3:
    xx=Xcentered[:,0]
    yy=Xcentered[:,1]
    zz=Xcentered[:,2]
    fig=plt.figure(1, figsize=(6, 5)) 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz)
    ax.set_title('ИСХОДНАЯ ИНФОРМАЦИЯ (отцентрированная)')
    ax.set_xlabel('первый признак',fontsize = 6)
    ax.set_ylabel('второй признак',fontsize = 6)
    ax.zaxis.set_label_text('третий признак',fontsize = 6)
    
elif base_feat == 2:
    xx=Xcentered[:,0]
    yy=Xcentered[:,1]
    fig=plt.figure(2)
    ax = fig.add_subplot(111)
    ax.scatter(xx, yy)
    ax.grid()
    ax.set_title('ИСХОДНАЯ ИНФОРМАЦИЯ (отцентрированная)')
    ax.set_xlabel('первый признак')
    ax.set_ylabel('второй признак')

covmat = np.cov(Xcentered.T) #транспонируем для вычисления ковариации
print('Ковариационная матрица')
print(covmat)

w, v = np.linalg.eig(covmat) #получаем собственные числа и векторы

sort_v = []
values = np.array(w)
for l in range(base_feat):
    col_v = []
    index_max = np.argmax(values)
    for k in range(base_feat):        
        col_v.append(v[k][index_max])
        values[index_max] = 0 #обнуляем максимальное собственное число, что бы оно не повторилось
    sort_v.append(col_v) #добавили вектор  с максимальным значение в конец массива
sort_v=np.array(sort_v).T

w=sorted(w)
w.reverse()
w = np.array(w)
print("Массив собственных чисел") 
print(w)
print("Массив собственных векторов") #столбец - это вектор
print(sort_v)

sum_w = w.sum() #Сумма дисперсий
proc_inf = w[:]/sum_w
print("Сколько % информации объясняет каждый признак: \n", proc_inf*100) #в процентах

dec_w = [] #уменьшенный массив дисперсий 
dec_v = [] #уменьшенный массив собственных векторов
dec_proc_inf = [] #уменьшенный массив объясняемой информации
for k in range(base_feat):
    v_row = []
    for l in range(dec_feat):
        v_row.append(v[k][l])
    dec_v.append(v_row)
dec_v=np.array(dec_v)

for l in range(dec_feat):
    dec_w.append(w[l])
    dec_proc_inf.append(proc_inf[l])
dec_w = np.array(dec_w)       
dec_proc_inf = np.array(dec_proc_inf)
X_transf = np.dot(Xcentered, dec_v) 
print("Уменьшенная матрица дисперсий")
print(dec_w)
print("Матрица на компонентах")
print(X_transf)

if  dec_feat == 3:
    xx=X_transf[:,0]
    yy=X_transf[:,1]
    zz=X_transf[:,2]
    fig=plt.figure(3, figsize=(6, 5)) 
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz)
    ax.set_title('ИНФОРМАЦИЯ ПОСЛЕ PCA')
    ax.set_xlabel('первая компонента',fontsize = 6)
    ax.set_ylabel('вторая компонента',fontsize = 6)
    ax.zaxis.set_label_text('третья компонента',fontsize = 6)
    
elif dec_feat == 2:
    xx=X_transf[:,0]
    yy=X_transf[:,1]
    fig=plt.figure(4)
    ax = fig.add_subplot(111)
    ax.scatter(xx, yy)
    ax.grid()
    ax.set_title('ИНФОРМАЦИЯ ПОСЛЕ PCA')
    ax.set_xlabel('первая компонента')
    ax.set_ylabel('вторая компонента')

n = random.randint(0,num_obj-1)
print("Восстановление элемента №  ", n)
Xrestored = np.dot( dec_v,X_transf[n]) + m
print("Восстановленное:")
print(Xrestored)
print("Оригинальное:")
print(X[n,:])

print('Матрица после снижения размерности методом PCA: \n', X_transf)
print('Сколько описывают главные компоненты ', dec_proc_inf)
print('Сколько сохранилось информации всего ', dec_proc_inf.sum())

