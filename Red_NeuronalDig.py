# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:15:23 2023

@author: emivi
"""
#importamos librerias
import mnist_loader, network
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
#import network

#obtencion de las imagenes 

training_data, validation_data, test_data = mnist_loader.load_data() #llamamos los datos 
p1=validation_data[0][20] #extraemos el primero de las imagenes de entrenamiento
print("el valor de la imagen es: ",validation_data[1][20]) #imprimimos el valor de la etiqueta
im=np.reshape(p1,(28,28)) #la acomodamos para visualizar
plt.imshow(im) # la visualizamos
imagnp=imread("no5.jpg") #imagen de prueba
plt.imshow(imagnp) #visualizar la imagen
plt.show()

#procesar la imagen de mnist
p1=np.reshape(im,(784,1))
net=network.Network([784,30,10]) #llamamos nuestra red

a=net.feedforward(p1) #metemos la imagen mnist
max = np.where(a ==np.amax(a)) #analizamos la respuesta
print("la red reconoce la imagen con el dígito", max[0])

#procesar la imagen nueva
imagnp=np.reshape(imagnp,(784,3)) #tomamos la imagen como vector y procesamos
#Convertimos a blanco y negro la imagen:
lst = []
for i in imagnp:
    pix=i[0]*0.2125+i[1]*0.7174+i[2]*0.0721 #transfomamos a escala de grises
    if(pix<125):
        pix=255. #Como la hoja es blanca y el papel negro, lo negro lo ponemos con mayor luminosidad
    else:
        pix=0. #lo blanco lo ponemos como negro
    lst.append(pix)
imagnp=np.array(lst).reshape(28,28) #acomodamos la imagen para poder ver como quedó
imagnp=(imagnp/imagnp.max()) #normalizamos
plt.imshow(imagnp) #mostramos la imagen en blanco y negro, y la vizualizamos
plt.show()

print("el valor de la imagen es 5")
imagenp=np.reshape(imagnp,(784,1))
a1=net.feedforward(imagenp)
max = np.where(a1 ==np.amax(a1))
print("la red reconoce la imagen con el dígito", max[0])
