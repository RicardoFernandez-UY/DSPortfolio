#regresiones lineales 
"""
The first line of the preceding code enables matplotlib to display the graphical output of the code in the notebook environment. 
The lines of code that follow use the import keyword to load various Python modules into our programming environment.
The last statement is used to set the aesthetic look of the graphs that matplotlib generates to the type displayed by the seaborn module.
"""
%matplotlib inline 
import matplotlib as mpl 
import seaborn as sns 
import matplotlib.pyplot as plt 
import statsmodels.formula.api as smf 
import statsmodels.graphics.api as smg 
import pandas as pd 
import numpy as np 
import patsy 
from statsmodels.graphics.correlation import plot_corr 
from sklearn.model_selection import train_test_split 
plt.style.use('seaborn') 

rawBostonData = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter02/Dataset/Boston.csv')

  rawBostonData.head()

#limpiamos los valores nulos
rawBostonData = rawBostonData.dropna()

#limpiamos duplicados
rawBostonData = rawBostonData.dropna()

list(rawBostonData.columns)

renamedBostonData = rawBostonData.rename(columns = {'CRIM':'crimeRatePerCapita', 
 ' ZN ':'landOver25K_sqft', 
 'INDUS ':'non-retailLandProptn', 
 'CHAS':'riverDummy', 
 'NOX':'nitrixOxide_pp10m', 
 'RM':'AvgNo.RoomsPerDwelling', 
 'AGE':'ProptnOwnerOccupied', 
 'DIS':'weightedDist', 
 'RAD':'radialHighwaysAccess', 
 'TAX':'propTaxRate_per10K', 
 'PTRATIO':'pupilTeacherRatio', 
 'LSTAT':'pctLowerStatus', 
 'MEDV':'medianValue_Ks'}) 
renamedBostonData.head() 

renamedBostonData.info()

#calculamos estadisticas basicas
"""
We used the pandas function, describe, called on the DataFrame to calculate simple statistics for numeric fields (this includes any field with a numpy number
in the DataFrame. The statistics include the minimum, the maximum, the count of rows in each column, the average of each column (mean), the 25th percentile, 
the 50th percentile, and the 75th percentile.  We transpose (using the .T function) the output of the describe function to get a better layout.
"""
renamedBostonData.describe(include=[np.number]).T

#Divide the DataFrame into training and test sets,
"""
We choose a test data size of 30%, which is 0.3. The train_test_split function is used to achieve this. We set the seed of the random number generator 
so that we can obtain a reproducible split each time we run this code. An arbitrary value of 10 is used here. 
It is good model-building practice to divide a dataset being used to develop a model into at least two parts. 
One part is used to develop the model and it is called a training set (X_train and y_train combined).
"""
X = renamedBostonData.drop('crimeRatePerCapita', axis = 1)
y = renamedBostonData[['crimeRatePerCapita']]
seed = 10
test_data_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_data_size, random_state = seed)
train_data = pd.concat([X_train, y_train], axis = 1)
test_data = pd.concat([X_test, y_test], axis = 1)

#Calculate and plot a correlation matrix for the train_data set
#el backslash es para continuar la linea en el renglon siguiente
corrMatrix = train_data.corr(method = 'pearson')
xnames=list(train_data.columns)
ynames=list(train_data.columns)
plot_corr(corrMatrix, xnames=xnames, ynames=ynames,\
          title=None, normcolor=False, cmap='RdYlBu_r')

"""
In the preceding heatmap, we can see that there is a strong positive correlation (an increase in one causes an increase in the other) 
between variables that have orange or red squares. 
There is a strong negative correlation (an increase in one causes a decrease in the other) between variables with blue squares. 
There is little or no correlation between variables with pale-colored squares.
"""


train_data.corr (method = 'pearson')

#usando Scatter graphs
"""
Use the subplots function in matplotlib to define a canvas (assigned the variable name fig in the following code) and a graph object (assigned the variable 
name ax in the following code) in Python. 
You can set the size of the graph by setting the figsize (width = 10, height = 6) argument of the function

Use the seaborn function regplot to create the scatter plot

The function accepts arguments for the independent variable (x), the dependent variable (y), the confidence interval of the regression parameters (ci), 
which takes values from 0 to 100, the DataFrame that has x and y (data), a matplotlib graph object (ax), and others to control the aesthetics of the 
points on the graph.  (In this case, the confidence interval is set to None
"""
fig, ax = plt.subplots(figsize=(10, 6))
fig.tight_layout()
ax.set_ylabel('Crime rate per Capita', fontsize=15, fontname='DejaVu Sans')
ax.set_xlabel("Median value of owner-occupied homes in $1000's", fontsize=15, fontname='DejaVu Sans')
ax.set_xlim(left=None, right=55)
ax.set_ylim(bottom=-10, top=30)
ax.tick_params(axis='both', which='major', labelsize=12)

sns.regplot(x='medianValue_Ks', y='crimeRatePerCapita', ci=None, data=train_data, ax=ax, color='k', scatter_kws={"s": 20,"color": "royalblue", "alpha":1})


fig, ax = plt.subplots(figsize=(10, 6))

