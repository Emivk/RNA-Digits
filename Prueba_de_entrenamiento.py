# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:32:55 2023

@author: emivi
"""
#cargamos librerias, mnist para cargar la base de datos y network que es
#el conjunto de funciones de una RNA densamente conectada simple
import mnist_loader, network

#separamos las entradas de la base de datos (mnist) en 3: 
# estos datos son del tipo zip
# datos de entrenamiento, para encontrar el arreglo de ws y bs optima para 
# la red, el proceso de entrenamiento 
# datos de validación, reservada para evaluar nuestro entrenamiento
#datos de prueba, la cual usamos para calificar la eficacia de la red

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data) #extraemos los datos en forma de lista
p1=training_data[0][0] #extraemos los datos de prueba
#así usamos las 784 entradas de los pixeles de la imagen y su etiqueta 
#correspondiente a cada imagen de forna [(pixeles,dígito),(pixeles,dígito),...]

#llamamos la func Network de la librería para darle estructura a la red 
net = network.Network([784, 30, 10]) #no de neuronas en la capa de entrada
                                     #escondida y de salida

#usamos la funcion Stochastic Gradient Descent para empezar el proceso
#de entrenamiento de la red
#arg: datos de entrenamiento, no. de epocas, tamaño del mini-bach, datos de prueba
net.SGD(training_data, 30, 10, 0.01, test_data=test_data)


