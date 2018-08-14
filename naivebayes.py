import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.decomposition import PCA
import seaborn as sns


plt.rc("font", size = 14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


#Importar o dataset
data = pd.read_csv("titanic_train.csv")

#Converte os dados categóricos em numéricos.
data['Sex_cleaned'] = np.where(data['Sex'] == 'male', 0, 1)
data["Embarked_cleaned"]=np.where(data["Embarked"]=="S",0, np.where(data["Embarked"]=="C",1, np.where(data["Embarked"]=="Q",2,3)))

data = data[[
    "Survived",
    "Pclass",
    "Sex_cleaned",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked_cleaned"
]].dropna(axis = 0, how = 'any')


# Verifica a distribuição dos campos.
for campo in list(data.columns):
    sns.countplot(x=campo, data = data, palette='hls')
    plt.show()

#Divide o dataset em treinamento e teste.
X_train, X_test = train_test_split(data, test_size=0.25, random_state=int(time.time()))

#Instancia o modelo classificador.
gnb = GaussianNB()

#"Fita" o modelo aos dados
used_features = ["Pclass", "Sex_cleaned", "Age", "SibSp", "Parch", "Fare", "Embarked_cleaned"]
gnb.fit(X_train[used_features], X_train["Survived"])

#Gera a previsão a partir do modelo "fitado".
y_pred = gnb.predict(X_test[used_features])

# Print results
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%".format(X_test.shape[0],(X_test["Survived"] != y_pred).sum(), 100*(1-(X_test["Survived"] != y_pred).sum()/X_test.shape[0])))

# Usa Análise de Componentes Principais (PCA) para analisar graficamente os resultados. Como há mais de duas dimensões, fica difícil 'enxegar' os dados, então o PCA vem em nosso socorro.
X = data.iloc[:,1:]
y = data.iloc[:,0]
pca = PCA(n_components=2).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)

plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='Lives', s=2, color='navy')
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='Dies', s=2, color='darkorange')
plt.legend()
plt.title('Titanic Data Set\nFirst Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().set_aspect('equal')
plt.show()