# Fatality Analysis Reporting System (FARS)

https://www.openml.org/d/40672  
Detailing the Factors Behind Traffic Fatalities on our Roads - FARS is a nationwide census providing NHTSA, Congress and the American public yearly data regarding fatal injuries suffered in motor vehicle traffic crashes.


```python
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
%matplotlib inline

import seaborn as sns
```

### Load the data from openml


```python
from sklearn.datasets import fetch_openml
fars = fetch_openml(name='fars')
```


```python
fars_pd = pd.DataFrame(fars.data, columns=fars.feature_names)
fars_pd['target'] = fars.target
```

##### What categories are available?


```python
fars_pd.columns
```




    Index(['CASE_STATE', 'AGE', 'SEX', 'PERSON_TYPE', 'SEATING_POSITION',
           'RESTRAINT_SYSTEM-USE', 'AIR_BAG_AVAILABILITY/DEPLOYMENT', 'EJECTION',
           'EJECTION_PATH', 'EXTRICATION', 'NON_MOTORIST_LOCATION',
           'POLICE_REPORTED_ALCOHOL_INVOLVEMENT', 'METHOD_ALCOHOL_DETERMINATION',
           'ALCOHOL_TEST_TYPE', 'ALCOHOL_TEST_RESULT',
           'POLICE-REPORTED_DRUG_INVOLVEMENT', 'METHOD_OF_DRUG_DETERMINATION',
           'DRUG_TEST_TYPE', 'DRUG_TEST_RESULTS_(1_of_3)',
           'DRUG_TEST_TYPE_(2_of_3)', 'DRUG_TEST_RESULTS_(2_of_3)',
           'DRUG_TEST_TYPE_(3_of_3)', 'DRUG_TEST_RESULTS_(3_of_3)',
           'HISPANIC_ORIGIN', 'TAKEN_TO_HOSPITAL',
           'RELATED_FACTOR_(1)-PERSON_LEVEL', 'RELATED_FACTOR_(2)-PERSON_LEVEL',
           'RELATED_FACTOR_(3)-PERSON_LEVEL', 'RACE', 'target'],
          dtype='object')




```python
categorie_to_color = 'target'
```

#### Have a quick look


```python
fars_pd.sample(5)
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
      <th>CASE_STATE</th>
      <th>AGE</th>
      <th>SEX</th>
      <th>PERSON_TYPE</th>
      <th>SEATING_POSITION</th>
      <th>RESTRAINT_SYSTEM-USE</th>
      <th>AIR_BAG_AVAILABILITY/DEPLOYMENT</th>
      <th>EJECTION</th>
      <th>EJECTION_PATH</th>
      <th>EXTRICATION</th>
      <th>NON_MOTORIST_LOCATION</th>
      <th>POLICE_REPORTED_ALCOHOL_INVOLVEMENT</th>
      <th>METHOD_ALCOHOL_DETERMINATION</th>
      <th>ALCOHOL_TEST_TYPE</th>
      <th>ALCOHOL_TEST_RESULT</th>
      <th>POLICE-REPORTED_DRUG_INVOLVEMENT</th>
      <th>METHOD_OF_DRUG_DETERMINATION</th>
      <th>DRUG_TEST_TYPE</th>
      <th>DRUG_TEST_RESULTS_(1_of_3)</th>
      <th>DRUG_TEST_TYPE_(2_of_3)</th>
      <th>DRUG_TEST_RESULTS_(2_of_3)</th>
      <th>DRUG_TEST_TYPE_(3_of_3)</th>
      <th>DRUG_TEST_RESULTS_(3_of_3)</th>
      <th>HISPANIC_ORIGIN</th>
      <th>TAKEN_TO_HOSPITAL</th>
      <th>RELATED_FACTOR_(1)-PERSON_LEVEL</th>
      <th>RELATED_FACTOR_(2)-PERSON_LEVEL</th>
      <th>RELATED_FACTOR_(3)-PERSON_LEVEL</th>
      <th>RACE</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3960</th>
      <td>2.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>29.0</td>
      <td>19.0</td>
      <td>17.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1542</th>
      <td>0.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>96.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>29.0</td>
      <td>19.0</td>
      <td>15.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59944</th>
      <td>30.0</td>
      <td>57.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>11.0</td>
      <td>12.0</td>
      <td>3.0</td>
      <td>9.0</td>
      <td>2.0</td>
      <td>16.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>99.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>999.0</td>
      <td>5.0</td>
      <td>999.0</td>
      <td>5.0</td>
      <td>999.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>29.0</td>
      <td>19.0</td>
      <td>15.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>49173</th>
      <td>22.0</td>
      <td>99.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>22.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>96.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>29.0</td>
      <td>19.0</td>
      <td>11.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1715</th>
      <td>0.0</td>
      <td>18.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>9.0</td>
      <td>97.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>999.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>29.0</td>
      <td>19.0</td>
      <td>15.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.pairplot(fars_pd.sample(1000))
```




    <seaborn.axisgrid.PairGrid at 0x7f91128fd9b0>




![png](FARS_files/FARS_10_1.png)


### Let's try PCA (Principal Component Analysis) first

normally that works quite good for low dimensional data

### We gonna try 2d and 3d visualization


```python
#3d
pca3d = PCA(n_components=3)
result3d = pca3d.fit_transform(fars_pd.drop(categorie_to_color, axis=1),fars_pd[categorie_to_color])
results3d_pd = pd.DataFrame(result3d,columns=['x','y','z'])
results3d_pd[categorie_to_color] = fars_pd[categorie_to_color].astype('int')
```


```python
#2d
pca2d = PCA(n_components=2)
result2d = pca2d.fit_transform(fars_pd.drop(categorie_to_color, axis=1),fars_pd[categorie_to_color])
results2d_pd = pd.DataFrame(result2d,columns=['x','y'])
results2d_pd[categorie_to_color] = fars_pd[categorie_to_color].astype('int')
```

### Let's plot


```python
fig_pca = plt.figure(figsize=(20,10))
ax = fig_pca.add_subplot(121, projection='3d')
ax.scatter(results3d_pd['x'], results3d_pd['y'], results3d_pd['z'],c=results3d_pd[categorie_to_color],  alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3d PCA")
ax.title
ax2 = fig_pca.add_subplot(122)
ax2.scatter(results2d_pd['x'], results2d_pd['y'],c=results3d_pd[categorie_to_color], alpha=0.5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title("2d PCA")

fig_pca.suptitle("Comparison between 2d and 3d PCA \n Components-Ratio for 3d {}, Compenents-Ratio for 2d {}".format(pca2d.explained_variance_ratio_,pca3d.explained_variance_ratio_))
plt.show()
```


![png](FARS_files/FARS_16_0.png)


#### not too impressive ... tsne should do better


```python
sample_ds = fars_pd.sample(3000)
sample_ds = sample_ds.reset_index().drop("index",axis=1)
tsne = TSNE(n_components=2, verbose=1, perplexity=35, n_iter=5000,learning_rate=200 )
tsne_results = tsne.fit_transform(sample_ds)
tsne_results_pd = pd.DataFrame(tsne_results,columns=['x','y'])
tsne_results_pd[categorie_to_color] = sample_ds[categorie_to_color].astype('int')
```

    [t-SNE] Computing 106 nearest neighbors...
    [t-SNE] Indexed 3000 samples in 0.014s...
    [t-SNE] Computed neighbors for 3000 samples in 0.318s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 3000
    [t-SNE] Computed conditional probabilities for sample 2000 / 3000
    [t-SNE] Computed conditional probabilities for sample 3000 / 3000
    [t-SNE] Mean sigma: 5.682091
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 61.308640
    [t-SNE] KL divergence after 5000 iterations: 0.619058



```python
sample_ds_3d = fars_pd.sample(3000)
sample_ds_3d = sample_ds_3d.reset_index().drop("index",axis=1)
tsne3d = TSNE(n_components=3, verbose=1, perplexity=45, n_iter=2500,learning_rate=200,)
tsne_results3d = tsne3d.fit_transform(sample_ds_3d)
tsne_results_pd3d = pd.DataFrame(tsne_results3d,columns=['x','y','z'])
tsne_results_pd3d[categorie_to_color] = sample_ds_3d[categorie_to_color].astype('int')
```

    [t-SNE] Computing 136 nearest neighbors...
    [t-SNE] Indexed 3000 samples in 0.007s...
    [t-SNE] Computed neighbors for 3000 samples in 0.311s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 3000
    [t-SNE] Computed conditional probabilities for sample 2000 / 3000
    [t-SNE] Computed conditional probabilities for sample 3000 / 3000
    [t-SNE] Mean sigma: 6.371458
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 59.635605
    [t-SNE] KL divergence after 2500 iterations: 0.496149



```python
fig_tsne = plt.figure(figsize=(20,10))
ax3 = fig_tsne.add_subplot(121)
im = ax3.scatter(tsne_results_pd['x'], tsne_results_pd['y'],c=tsne_results_pd[categorie_to_color],alpha=0.3, cmap=plt.cm.tab10_r)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_title("2d tSNE")
fig_tsne.colorbar(im, orientation="horizontal", pad=0.2)

ax4 = fig_tsne.add_subplot(122, projection='3d')
im2 = ax4.scatter(tsne_results_pd3d['x'], tsne_results_pd3d['y'],tsne_results_pd3d['z'],c=tsne_results_pd3d[categorie_to_color],alpha=0.3, cmap=plt.cm.tab10_r)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title("3d tSNE")


fig_tsne.colorbar(im2, orientation="horizontal", pad=0.2)

# cbaxes = fig_tsne.add_axes([0.8, 0.1, 0.03, 0.8]) 
# fig_tsne.colorbar(im2, cax=cbaxes)
plt.show()
```


![png](FARS_files/FARS_20_0.png)



```python
#ToDo
#Accelerate with RAPIDS and use all the Data
#Use different categorie to color graph
```


```python

```
