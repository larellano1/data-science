''' Exemplo retirado de https://datascienceplus.com/building-a-logistic-regression-in-python-step-by-step/ 
com pequenas adaptações'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

plt.rc("font", size = 14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Baixa os dados para a memória como um DataFrame e exclui as linhas vazias.
data = pd.read_csv("banking.csv")
data = data.dropna()


# Começa análise exploratória dos dados.
print(data.shape)
print(list(data.columns))


# Verifica a distribuição dos campos.
for campo in list(data.columns):
    sns.countplot(x=campo, data = data, palette='hls')
    plt.show()

# Verifica a existência de campos vazios.
print(data.isnull().sum())

# Elimina as colunas que não vamos precisar para a regressão.
data.drop(data.columns[[0, 3, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19]], axis=1, inplace=True)

# Cria dummies para cada uma das variáveis categóricas independentes que vamos usar no nosso modelo.
data2 = pd.get_dummies(data, columns =['job', 'marital', 'default', 'housing', 'loan', 'poutcome'])
data2.drop(data2.columns[[12, 16, 18, 21, 24]], axis=1, inplace=True)
print(data2.columns)

# Verifica a correlação entre as variáveis, para checar se são independentes, o que é condição para uso da regressão logística.
sns.heatmap(data2.corr())
plt.show()

# Divide os dados em subconjunto de treino e subconjunto de teste. Em seguida, imprime o tamanho do subconjunto de teste para avaliar se o tamanho é suficiente.
X = data2.iloc[:,1:]
y = data2.iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print(X_train.shape)

# Obtém um modelo de regressão logística 'fitando' os dados de teste. 
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

# Prevê os Ys de teste a partir dos Xs de teste com o modelo 'fitado'.
y_pred = classifier.predict(X_test)

# Imprime a matriz de confusão, que mostra a quantidade de erros e acertos para cada caso testado. Imprime a acurácia do teste.
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))

# Diagnóstico final sobre o modelo e resultados. Precision = capacidade de não dar falso positivo. Recall = capacidade de não dar falso negativo.
print(classification_report(y_test, y_pred))


# Usa Análise de Componentes Principais (PCA) para analisar graficamente os resultados. Como há mais de duas dimensões, fica difícil 'enxegar' os dados, então o PCA vem em nosso socorro.
X = data2.iloc[:,1:]
y = data2.iloc[:,0]
pca = PCA(n_components=2).fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(pca, y, random_state=0)

plt.figure(dpi=120)
plt.scatter(pca[y.values==0,0], pca[y.values==0,1], alpha=0.5, label='YES', s=2, color='navy')
plt.scatter(pca[y.values==1,0], pca[y.values==1,1], alpha=0.5, label='NO', s=2, color='darkorange')
plt.legend()
plt.title('Bank Marketing Data Set\nFirst Two Principal Components')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.gca().set_aspect('equal')
plt.show()