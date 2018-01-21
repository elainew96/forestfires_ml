import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle

from sklearn import decomposition
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

#csv file to pandas dataframe
df = pd.read_csv('forestfires.csv')

#one hot encoding
#days = ['mon','tue','wed','thu','fri','sat','sun']
#months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
#for i in days:
#    df[i] = np.where(df['day']==i,1,0)
#for i in months:
#    df[i] = np.where(df['month']==i,1,0)
del df['day']
del df['month']
#change area to logrithmic scaling
#df['area'] = math.log(float(df['area'])+1)
df['area'].apply(lambda x: math.log(x+1))
ff = df.copy()
y = ff['area']
x = ff.drop('area',axis=1)

print x
print y

raw_input("Press Enter: ")

# Load or fit regression model
try:
    pickle_y_rbf = open('y_data_rbf','rb')
    pickle_y_poly=open('y_data_poly','wb')
    y_rbf = pickle.load(pickle_y_rbf)
    y_poly = pickle.load(pickle_y_poly)
    pickle_y_rbf.close()
    pickle_y_poly.close()
except:
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    y_rbf = svr_rbf.fit(x, y).predict(x)
    y_poly = svr_poly.fit(x, y).predict(x)
    pickle_y_rbf = open('y_data_rbf','wb')
    pickle.dump(y_rbf,pickle_y_rbf)
    pickle_y_rbf.close()
    pickle_y_poly=open('y_data_poly','wb')
    pickle.dump(y_poly,pickle_y_poly)
    pickle_y_poly.close()

exit(0)

# Look at results
#lw = 2
#plt.scatter(x,y,facecolors='none',edgecolors='b',label='original data')
#plt.plot(x,y_rbf,color='navy',lw=lw,label='RBF model')
#plt.plot(x,y_poly,color='c',lw=lw,label='Polynomial model')
#plt.xlabel('data')
#plt.ylabel('target')
#plt.title('Support Vector Regression')
#plt.legend()
#plt.show()

# Error
rbf_error = mean_squared_error(y,y_rbf)
poly_error = mean_squared_error(y,y_poly)

print 'RBF model error: ' + str(rbf_error)
print 'Poly model error: ' + str(poly_error)

raw_input("Press Enter to continue to PCA...")

#plt.close()
#PCA method

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
