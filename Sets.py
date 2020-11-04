#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np
import random

#Entrega lista con los indices de las instancias de cierta categoria de objeto astronomico
def get_index_cat(dic_datos, cat):
    index_list = []
    for i in range(len(dic_datos['labels'])):
        if dic_datos['labels'][i]==cat:
            index_list.append(i)
    return index_list

# Genera listas de índices para train, validation y test
def get_sets_index(dic_datos):
    val_index=np.array([])
    test_index=np.array([])
    train_index=np.array([])
    for k in range (5):
        L=get_index_cat(dic_datos, k)
        random.shuffle(L)
        test_index=np.append(test_index,L[0:200])
        val_index=np.append(val_index,L[200:300])
        train_index=np.append(train_index,L[300:])
        random.shuffle(test_index)
        random.shuffle(val_index)
        random.shuffle(train_index)
    return train_index, val_index, test_index

#crea listas con los sets de datos
def build_sets(dic_datos):
    x_train = []
    x_val = []
    x_test = []
    y_train = []
    y_val = []
    y_test = []
    train_index, val_index, test_index = get_sets_index(dic_datos) #obtenemos los indices
    for i in range(len(train_index)):
        k=int(train_index[i])
        x_train.append(dic_datos['im_norm21'][k])
        y_train.append(dic_datos['labels'][k])
    for i in range(len(val_index)):
        k=int(val_index[i])
        x_val.append(dic_datos['im_norm21'][k])
        y_val.append(dic_datos['labels'][k])
    for i in range(len(test_index)):
        k=int(test_index[i])
        x_test.append(dic_datos['im_norm21'][k])
        y_test.append(dic_datos['labels'][k])
    #x_train = np.array(x_train)
    #x_val = np.array(x_val)
    #x_test = np.array(x_test)
    #y_train = np.array(y_train)
    #y_val = np.array(y_val)
    #y_test = np.array(y_test)
    return x_train, x_val, x_test, y_train, y_val, y_test