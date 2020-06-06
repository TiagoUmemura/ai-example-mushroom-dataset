from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
import pandas as pd

#Converte csv para array de arrays
mushrooms_data = np.genfromtxt("mushrooms.csv", dtype=None, delimiter=',', names=True)

rotulos = [] #lista com rotulosdo arquivo inteiro
caracteristicas = [] #lista com as caracteristicas

#converter cada letra para um numero
for item in mushrooms_data:
	if item[0] == "p": #primeira posicao do array de cada caracteristica e o rotulo
		rotulos.append(1) # se p rotulo 1
	else:
		rotulos.append(0) # se e rotulo 0
	

	caracteristica = item.tolist()[1:] # da segunda posicao pre frente sao caracteristicas
	caracteristicaArray = [] # array com as caracteristicas convertidas para numero

	for x in range(0, len(caracteristica)):
		if caracteristica[x] == "b":
			caracteristicaArray.append(0)
		elif caracteristica[x] == "c":
			caracteristicaArray.append(1)
		elif caracteristica[x] == "x":
			caracteristicaArray.append(2)
		elif caracteristica[x] == "f":
			caracteristicaArray.append(3)
		elif caracteristica[x] == "k":
			caracteristicaArray.append(4)
		elif caracteristica[x] == "s":
			caracteristicaArray.append(5)
		elif caracteristica[x] == "g":
			caracteristicaArray.append(6)
		elif caracteristica[x] == "y":
			caracteristicaArray.append(7)
		elif caracteristica[x] == "s":
			caracteristicaArray.append(8)
		elif caracteristica[x] == "n":
			caracteristicaArray.append(9)
		elif caracteristica[x] == "r":
			caracteristicaArray.append(10)
		elif caracteristica[x] == "p":
			caracteristicaArray.append(11)
		elif caracteristica[x] == "u":
			caracteristicaArray.append(12)
		elif caracteristica[x] == "e":
			caracteristicaArray.append(13)
		elif caracteristica[x] == "w":
			caracteristicaArray.append(14)
		elif caracteristica[x] == "l":
			caracteristicaArray.append(15)
		elif caracteristica[x] == "a":
			caracteristicaArray.append(16)
		elif caracteristica[x] == "m":
			caracteristicaArray.append(17)
		elif caracteristica[x] == "n":
			caracteristicaArray.append(18)
		elif caracteristica[x] == "h":
			caracteristicaArray.append(19)
		elif caracteristica[x] == "o":
			caracteristicaArray.append(20)
		elif caracteristica[x] == "t":
			caracteristicaArray.append(21)
		elif caracteristica[x] == "z":
			caracteristicaArray.append(22)
		elif caracteristica[x] == "?":
			caracteristicaArray.append(23)
		elif caracteristica[x] == "v":
			caracteristicaArray.append(24)
		elif caracteristica[x] == "d":
			caracteristicaArray.append(25)


	caracteristicas.append(caracteristicaArray)
###################################################
#print(rotulos)
#print(caracteristicas)

treino_rotulos = rotulos[0:4000]
treino_caracteristicas = caracteristicas[0:4000]

teste_rotulos = rotulos[4001:8000]
teste_caracteristicas = caracteristicas[4001:8000]

#clf = svm.SVC()

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(treino_caracteristicas,treino_rotulos)

#for caracteristica in teste_caracteristicas:
#	print(clf.predict(caracteristica))
print("Random Forest")
print(clf.score(teste_caracteristicas,teste_rotulos))

# Executando classificador random forest com 10, 100 e 500 arvores, 10 vezes cada uma
for j in 10, 100, 500:
	print ("Numero de arvores: " + str(j))
	soma = 0
	for i in range(0,10):
		clf = RandomForestClassifier(n_estimators=j)
		clf = clf.fit(treino_caracteristicas,treino_rotulos)
		print(clf.score(teste_caracteristicas,teste_rotulos))
		soma += clf.score(teste_caracteristicas,teste_rotulos)
	#print(soma)
	media = soma/10
	print("Media: " + str(media))


featureImp = clf.feature_importances_

print(featureImp)
print("posicao odor: " + str(featureImp[4]))

clf2 = svm.SVC()
clf2.fit(treino_caracteristicas, treino_rotulos)
print("SVM")
print(clf2.score(teste_caracteristicas, teste_rotulos))
#print(clf2.support_vectors_)

clf3 = NearestCentroid()
clf3.fit(treino_caracteristicas, treino_rotulos)
print("KNN Centroide")
print(clf3.score(teste_caracteristicas, teste_rotulos))
#print(clf3.centroids_)

