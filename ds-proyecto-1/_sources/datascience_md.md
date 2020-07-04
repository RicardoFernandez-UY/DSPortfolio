# Mi cuaderno Python
* This is a bullet list
* This is a bullet list
* This is a bullet list



```python
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
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    


```python
rawBostonData = pd.read_csv('https://raw.githubusercontent.com/PacktWorkshops/The-Data-Science-Workshop/master/Chapter02/Dataset/Boston.csv')
```


```python
  rawBostonData.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
#limpiamos los valores nulos
rawBostonData = rawBostonData.dropna()
```


```python
#limpiamos duplicados
rawBostonData = rawBostonData.dropna()
```


```python
list(rawBostonData.columns)
```




    ['CRIM',
     ' ZN ',
     'INDUS ',
     'CHAS',
     'NOX',
     'RM',
     'AGE',
     'DIS',
     'RAD',
     'TAX',
     'PTRATIO',
     'LSTAT',
     'MEDV']




```python
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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crimeRatePerCapita</th>
      <th>landOver25K_sqft</th>
      <th>non-retailLandProptn</th>
      <th>riverDummy</th>
      <th>nitrixOxide_pp10m</th>
      <th>AvgNo.RoomsPerDwelling</th>
      <th>ProptnOwnerOccupied</th>
      <th>weightedDist</th>
      <th>radialHighwaysAccess</th>
      <th>propTaxRate_per10K</th>
      <th>pupilTeacherRatio</th>
      <th>pctLowerStatus</th>
      <th>medianValue_Ks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1</td>
      <td>296</td>
      <td>15.3</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2</td>
      <td>242</td>
      <td>17.8</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3</td>
      <td>222</td>
      <td>18.7</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
renamedBostonData.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 506 entries, 0 to 505
    Data columns (total 13 columns):
     #   Column                  Non-Null Count  Dtype  
    ---  ------                  --------------  -----  
     0   crimeRatePerCapita      506 non-null    float64
     1   landOver25K_sqft        506 non-null    float64
     2   non-retailLandProptn    506 non-null    float64
     3   riverDummy              506 non-null    int64  
     4   nitrixOxide_pp10m       506 non-null    float64
     5   AvgNo.RoomsPerDwelling  506 non-null    float64
     6   ProptnOwnerOccupied     506 non-null    float64
     7   weightedDist            506 non-null    float64
     8   radialHighwaysAccess    506 non-null    int64  
     9   propTaxRate_per10K      506 non-null    int64  
     10  pupilTeacherRatio       506 non-null    float64
     11  pctLowerStatus          506 non-null    float64
     12  medianValue_Ks          506 non-null    float64
    dtypes: float64(10), int64(3)
    memory usage: 55.3 KB
    

## Algunas estadisticas


```python
#calculamos estadisticas basicas
"""
We used the pandas function, describe, called on the DataFrame to calculate simple statistics for numeric fields (this includes any field with a numpy number
in the DataFrame. The statistics include the minimum, the maximum, the count of rows in each column, the average of each column (mean), the 25th percentile, 
the 50th percentile, and the 75th percentile.  We transpose (using the .T function) the output of the describe function to get a better layout.
"""
renamedBostonData.describe(include=[np.number]).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>crimeRatePerCapita</th>
      <td>506.0</td>
      <td>3.613524</td>
      <td>8.601545</td>
      <td>0.00632</td>
      <td>0.082045</td>
      <td>0.25651</td>
      <td>3.677082</td>
      <td>88.9762</td>
    </tr>
    <tr>
      <th>landOver25K_sqft</th>
      <td>506.0</td>
      <td>11.363636</td>
      <td>23.322453</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>12.500000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>non-retailLandProptn</th>
      <td>506.0</td>
      <td>11.136779</td>
      <td>6.860353</td>
      <td>0.46000</td>
      <td>5.190000</td>
      <td>9.69000</td>
      <td>18.100000</td>
      <td>27.7400</td>
    </tr>
    <tr>
      <th>riverDummy</th>
      <td>506.0</td>
      <td>0.069170</td>
      <td>0.253994</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>nitrixOxide_pp10m</th>
      <td>506.0</td>
      <td>0.554695</td>
      <td>0.115878</td>
      <td>0.38500</td>
      <td>0.449000</td>
      <td>0.53800</td>
      <td>0.624000</td>
      <td>0.8710</td>
    </tr>
    <tr>
      <th>AvgNo.RoomsPerDwelling</th>
      <td>506.0</td>
      <td>6.284634</td>
      <td>0.702617</td>
      <td>3.56100</td>
      <td>5.885500</td>
      <td>6.20850</td>
      <td>6.623500</td>
      <td>8.7800</td>
    </tr>
    <tr>
      <th>ProptnOwnerOccupied</th>
      <td>506.0</td>
      <td>68.574901</td>
      <td>28.148861</td>
      <td>2.90000</td>
      <td>45.025000</td>
      <td>77.50000</td>
      <td>94.075000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>weightedDist</th>
      <td>506.0</td>
      <td>3.795043</td>
      <td>2.105710</td>
      <td>1.12960</td>
      <td>2.100175</td>
      <td>3.20745</td>
      <td>5.188425</td>
      <td>12.1265</td>
    </tr>
    <tr>
      <th>radialHighwaysAccess</th>
      <td>506.0</td>
      <td>9.549407</td>
      <td>8.707259</td>
      <td>1.00000</td>
      <td>4.000000</td>
      <td>5.00000</td>
      <td>24.000000</td>
      <td>24.0000</td>
    </tr>
    <tr>
      <th>propTaxRate_per10K</th>
      <td>506.0</td>
      <td>408.237154</td>
      <td>168.537116</td>
      <td>187.00000</td>
      <td>279.000000</td>
      <td>330.00000</td>
      <td>666.000000</td>
      <td>711.0000</td>
    </tr>
    <tr>
      <th>pupilTeacherRatio</th>
      <td>506.0</td>
      <td>18.455534</td>
      <td>2.164946</td>
      <td>12.60000</td>
      <td>17.400000</td>
      <td>19.05000</td>
      <td>20.200000</td>
      <td>22.0000</td>
    </tr>
    <tr>
      <th>pctLowerStatus</th>
      <td>506.0</td>
      <td>12.653063</td>
      <td>7.141062</td>
      <td>1.73000</td>
      <td>6.950000</td>
      <td>11.36000</td>
      <td>16.955000</td>
      <td>37.9700</td>
    </tr>
    <tr>
      <th>medianValue_Ks</th>
      <td>506.0</td>
      <td>22.532806</td>
      <td>9.197104</td>
      <td>5.00000</td>
      <td>17.025000</td>
      <td>21.20000</td>
      <td>25.000000</td>
      <td>50.0000</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```

### Algunas Grafiquitas


```python
#Calculate and plot a correlation matrix for the train_data set
#el backslash es para continuar la linea en el renglon siguiente
corrMatrix = train_data.corr(method = 'pearson')
xnames=list(train_data.columns)
ynames=list(train_data.columns)
plot_corr(corrMatrix, xnames=xnames, ynames=ynames,\
          title=None, normcolor=False, cmap='RdYlBu_r')
```




![png](output_13_0.png)




![png](output_13_1.png)



```python
"""
In the preceding heatmap, we can see that there is a strong positive correlation (an increase in one causes an increase in the other) 
between variables that have orange or red squares. 
There is a strong negative correlation (an increase in one causes a decrease in the other) between variables with blue squares. 
There is little or no correlation between variables with pale-colored squares.
"""

```




    '\nIn the preceding heatmap, we can see that there is a strong positive correlation (an increase in one causes an increase in the other) \nbetween variables that have orange or red squares. \nThere is a strong negative correlation (an increase in one causes a decrease in the other) between variables with blue squares. \nThere is little or no correlation between variables with pale-colored squares.\n'




```python
train_data.corr (method = 'pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>landOver25K_sqft</th>
      <th>non-retailLandProptn</th>
      <th>riverDummy</th>
      <th>nitrixOxide_pp10m</th>
      <th>AvgNo.RoomsPerDwelling</th>
      <th>ProptnOwnerOccupied</th>
      <th>weightedDist</th>
      <th>radialHighwaysAccess</th>
      <th>propTaxRate_per10K</th>
      <th>pupilTeacherRatio</th>
      <th>pctLowerStatus</th>
      <th>medianValue_Ks</th>
      <th>crimeRatePerCapita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>landOver25K_sqft</th>
      <td>1.000000</td>
      <td>-0.540095</td>
      <td>-0.059189</td>
      <td>-0.520305</td>
      <td>0.355346</td>
      <td>-0.577457</td>
      <td>0.659340</td>
      <td>-0.311920</td>
      <td>-0.324172</td>
      <td>-0.424612</td>
      <td>-0.435827</td>
      <td>0.422574</td>
      <td>-0.198455</td>
    </tr>
    <tr>
      <th>non-retailLandProptn</th>
      <td>-0.540095</td>
      <td>1.000000</td>
      <td>0.065271</td>
      <td>0.758178</td>
      <td>-0.399166</td>
      <td>0.667887</td>
      <td>-0.728968</td>
      <td>0.580813</td>
      <td>0.702973</td>
      <td>0.398513</td>
      <td>0.607457</td>
      <td>-0.508338</td>
      <td>0.387471</td>
    </tr>
    <tr>
      <th>riverDummy</th>
      <td>-0.059189</td>
      <td>0.065271</td>
      <td>1.000000</td>
      <td>0.091469</td>
      <td>0.107996</td>
      <td>0.106329</td>
      <td>-0.098551</td>
      <td>0.022731</td>
      <td>-0.007864</td>
      <td>-0.094255</td>
      <td>-0.041110</td>
      <td>0.136831</td>
      <td>-0.044587</td>
    </tr>
    <tr>
      <th>nitrixOxide_pp10m</th>
      <td>-0.520305</td>
      <td>0.758178</td>
      <td>0.091469</td>
      <td>1.000000</td>
      <td>-0.306510</td>
      <td>0.742016</td>
      <td>-0.776311</td>
      <td>0.606721</td>
      <td>0.662164</td>
      <td>0.206809</td>
      <td>0.603656</td>
      <td>-0.453424</td>
      <td>0.405813</td>
    </tr>
    <tr>
      <th>AvgNo.RoomsPerDwelling</th>
      <td>0.355346</td>
      <td>-0.399166</td>
      <td>0.107996</td>
      <td>-0.306510</td>
      <td>1.000000</td>
      <td>-0.263085</td>
      <td>0.215439</td>
      <td>-0.183000</td>
      <td>-0.280341</td>
      <td>-0.350828</td>
      <td>-0.586573</td>
      <td>0.666761</td>
      <td>-0.167258</td>
    </tr>
    <tr>
      <th>ProptnOwnerOccupied</th>
      <td>-0.577457</td>
      <td>0.667887</td>
      <td>0.106329</td>
      <td>0.742016</td>
      <td>-0.263085</td>
      <td>1.000000</td>
      <td>-0.751059</td>
      <td>0.458717</td>
      <td>0.515376</td>
      <td>0.289976</td>
      <td>0.639881</td>
      <td>-0.419062</td>
      <td>0.355730</td>
    </tr>
    <tr>
      <th>weightedDist</th>
      <td>0.659340</td>
      <td>-0.728968</td>
      <td>-0.098551</td>
      <td>-0.776311</td>
      <td>0.215439</td>
      <td>-0.751059</td>
      <td>1.000000</td>
      <td>-0.494932</td>
      <td>-0.543333</td>
      <td>-0.259140</td>
      <td>-0.522120</td>
      <td>0.289658</td>
      <td>-0.378997</td>
    </tr>
    <tr>
      <th>radialHighwaysAccess</th>
      <td>-0.311920</td>
      <td>0.580813</td>
      <td>0.022731</td>
      <td>0.606721</td>
      <td>-0.183000</td>
      <td>0.458717</td>
      <td>-0.494932</td>
      <td>1.000000</td>
      <td>0.908578</td>
      <td>0.462290</td>
      <td>0.456592</td>
      <td>-0.383132</td>
      <td>0.608838</td>
    </tr>
    <tr>
      <th>propTaxRate_per10K</th>
      <td>-0.324172</td>
      <td>0.702973</td>
      <td>-0.007864</td>
      <td>0.662164</td>
      <td>-0.280341</td>
      <td>0.515376</td>
      <td>-0.543333</td>
      <td>0.908578</td>
      <td>1.000000</td>
      <td>0.462556</td>
      <td>0.528029</td>
      <td>-0.478903</td>
      <td>0.565035</td>
    </tr>
    <tr>
      <th>pupilTeacherRatio</th>
      <td>-0.424612</td>
      <td>0.398513</td>
      <td>-0.094255</td>
      <td>0.206809</td>
      <td>-0.350828</td>
      <td>0.289976</td>
      <td>-0.259140</td>
      <td>0.462290</td>
      <td>0.462556</td>
      <td>1.000000</td>
      <td>0.374842</td>
      <td>-0.503692</td>
      <td>0.276530</td>
    </tr>
    <tr>
      <th>pctLowerStatus</th>
      <td>-0.435827</td>
      <td>0.607457</td>
      <td>-0.041110</td>
      <td>0.603656</td>
      <td>-0.586573</td>
      <td>0.639881</td>
      <td>-0.522120</td>
      <td>0.456592</td>
      <td>0.528029</td>
      <td>0.374842</td>
      <td>1.000000</td>
      <td>-0.743548</td>
      <td>0.406340</td>
    </tr>
    <tr>
      <th>medianValue_Ks</th>
      <td>0.422574</td>
      <td>-0.508338</td>
      <td>0.136831</td>
      <td>-0.453424</td>
      <td>0.666761</td>
      <td>-0.419062</td>
      <td>0.289658</td>
      <td>-0.383132</td>
      <td>-0.478903</td>
      <td>-0.503692</td>
      <td>-0.743548</td>
      <td>1.000000</td>
      <td>-0.378949</td>
    </tr>
    <tr>
      <th>crimeRatePerCapita</th>
      <td>-0.198455</td>
      <td>0.387471</td>
      <td>-0.044587</td>
      <td>0.405813</td>
      <td>-0.167258</td>
      <td>0.355730</td>
      <td>-0.378997</td>
      <td>0.608838</td>
      <td>0.565035</td>
      <td>0.276530</td>
      <td>0.406340</td>
      <td>-0.378949</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
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

```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f66beb36940>




![png](output_16_1.png)

