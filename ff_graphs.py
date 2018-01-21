import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import decomposition

#csv file to pandas dataframe
df = pd.read_csv('forestfires.csv')

#one hot encoding??
days = ['mon','tue','wed','thu','fri','sat','sun']
months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
print df
#for i in days:
#    df[i] = np.where(df['day']==i,1,0)
#for i in months:
#    df[i] = np.where(df['month']==i,1,0)
del df['day']
del df['month']

pca = decomposition.PCA(n_components=2)
data_mat = pca.fit_transform(df)
print pca.components_

#fit transform by hand:
#transform = [] #contains all of the vectors
#for i in range(len(data_mat)):
    #transform.append(tuple(pca.components_.dot(data_mat[i].transpose())))
#print pca.components_.dot(data_mat[0].transpose()) #vector that has been projected

#plot the vectors
x = list(data_mat.transpose()[0])
y = list(data_mat.transpose()[1]) #
plt.scatter(x,y,facecolors='none',s=40,edgecolors='b')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()
#see that area, DMC, DC, and ISI are good ways of classifying fires
