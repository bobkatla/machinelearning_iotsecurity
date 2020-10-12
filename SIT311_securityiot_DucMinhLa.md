# <span style="color:#0b486b">SIT311 - SE3: Designing User-centric IoT Applications</span>


## <span style="color:#0b486b">Research implementation: machine learning techiniques for IoT security purpose</span>

**Authors:**

Duc Minh La - 217616227

**Outline:** 

1. Set up and data preparation
2. Methodology
3. Results and discussion
4. Reference

### <span style="color:#0b486b">1. Set up and data preparation: </span>
#### <span style="color:#0b486b">1.1. Acknowledgement and introduction </span>
This report is a continue from the previous research paper reagrding the machine learning solutions for IoT security. The research report have touched many different solutions for different security aspect such as authorization or malware dectection. This report will discuss and create an implementation of the malware detection solution using classification methods to distinguish between benign data and malicious data to detect viruses or malwares in the future.

The used datases are progressed data from an open source[1] with the work of [2].

#### <span style="color:#0b486b">1.2. Set up the work place </span>


```python
#import needed libraries
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
%matplotlib inline

# import the classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

# import the checking method
from sklearn.model_selection import KFold, StratifiedKFold, ShuffleSplit
from sklearn.model_selection import cross_val_score

# import the method for selecting best features
from sklearn.feature_selection import SelectKBest, mutual_info_classif
```

#### <span style="color:#0b486b">1.3. Data preparation </span>
From the given source, there are basically 2 data file, 1 is benign data and other is the malacious data. To perform machine leaning classifications, the 2 dataset would be combined into 1 dataset with an extra feature or column named "auth" that show 1 for benign data and 0 for malacious data. 


```python
#import the dataframe from csv file
df1_o = pd.read_csv('combo.csv') # the malacious data
df2_o = pd.read_csv('benign.csv') # the benign data
#check the size of the dataset
print('*the shape of the malacious dataframe is: {}'.format(df1_o.shape))
print('*the shape of the benign dataframe is: {}'.format(df2_o.shape))
```

    *the shape of the malacious dataframe is: (59718, 115)
    *the shape of the benign dataframe is: (49548, 115)
    

As can be seen above, the size of the 2 given dataset are extremly large, performing machine learning methods on the combined dataset would be extremely long [3]. This in real life scenario is understandable and accepted for the best result, plus with better computation resources such as cloud, the time can be reduced. However, for the sake of demonstration in this report and also because of the limitation of hardware, the tested would be reduced to only 2500 record from each dataset, hence, 5000 records to feed the machine learning algorithms.


```python
df1 = df1_o.head(n = 2500)
df2 = df2_o.head(n = 2500)

print('*the new shape of the malacious dataframe is: {}'.format(df1.shape))
print('*the new shape of the benign dataframe is: {}'.format(df2.shape))
```

    *the new shape of the malacious dataframe is: (2500, 115)
    *the new shape of the benign dataframe is: (2500, 115)
    

To continue, the dataset would go through data cleaning by dropping all the null values and then combined to create the dataset for machine learning


```python
# dropping null values in both datasets - deletewise
mala = df1.dropna()
beni = df2.dropna() 

# combined 2 dataset to create the final dataset 
frames = [mala, beni]
X = pd.concat(frames)
print('*the shape of the combined dataframe is: {}'.format(X.shape))

# getting the target array of auth
mala['auth'] = 1
beni['auth'] = 0
frames2 = [mala, beni]
combine = pd.concat(frames2)
y = combine['auth']
```

    *the shape of the combined dataframe is: (5000, 115)
    

### <span style="color:#0b486b">2. Methodology: </span>
As discussed above, the dataset is extremely large and for demonstration in this report, the dataset would be reduced. Though the number of records have been reduced, the number of features is still extremly high with the number of 115 features. The approach for reducing the number of features would be calculating the mutual information scores (FMI) for each features and choose the best features for machines learning. This is also a great method for improving the computating time of machine learning algorithms in real life, hence, different number of features would be used to test for the accuracy.
#### <span style="color:#0b486b">2.1. Calculating FMI scores </span>


```python
feature_names = X.columns #get the feature names of later

# calculating the FMI for each features in the d
FMI = mutual_info_classif(X, y, random_state=20)
# print("Features Mutual Info Scores: {}".format(FMI))

# connecting the calculated scores with the corresponding features
di = {}
for i, items in enumerate(FMI):
    di[feature_names[i]] = items
# print(di)

# sorted the features
sorted_d = sorted(di.items(), key=lambda x: x[1], reverse=True)
# print(sorted_d)

# putting the sorted features into an array
ranked_ls = []
for items in sorted_d:
    ranked_ls.append(items[0])
# print(ranked_ls)

# create a table/dataframe for displaying the best features
ranked_df = pd.DataFrame()
ranked_df['features'] = ranked_ls
ranked_df['FMI'] = sorted(FMI, reverse=True)
ranked_df['ranked']=ranked_df['FMI'].rank(ascending=0)

ranked_df
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
      <th>features</th>
      <th>FMI</th>
      <th>ranked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>H_L0.01_weight</td>
      <td>0.690647</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MI_dir_L0.01_weight</td>
      <td>0.690453</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HH_L0.01_magnitude</td>
      <td>0.687814</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MI_dir_L0.01_mean</td>
      <td>0.687397</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>H_L0.01_mean</td>
      <td>0.687397</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>110</th>
      <td>HpHp_L3_pcc</td>
      <td>0.000000</td>
      <td>111.5</td>
    </tr>
    <tr>
      <th>111</th>
      <td>HpHp_L1_radius</td>
      <td>0.000000</td>
      <td>111.5</td>
    </tr>
    <tr>
      <th>112</th>
      <td>HpHp_L1_covariance</td>
      <td>0.000000</td>
      <td>111.5</td>
    </tr>
    <tr>
      <th>113</th>
      <td>HpHp_L1_pcc</td>
      <td>0.000000</td>
      <td>111.5</td>
    </tr>
    <tr>
      <th>114</th>
      <td>HpHp_L0.1_pcc</td>
      <td>0.000000</td>
      <td>111.5</td>
    </tr>
  </tbody>
</table>
<p>115 rows × 3 columns</p>
</div>



From the list above, the report will test and comparing the results from 50, 25, 10 and 5 features


```python
print('the top 50 features are: ')
print(list(ranked_df.head(n=50)['features']))
print()
print('the top 25 features are: ')
print(list(ranked_df.head(n=25)['features']))
print()
print('the top 10 features are: ')
print(list(ranked_df.head(n=10)['features']))
print()
print('the top 5 features are: ')
print(list(ranked_df.head(n=5)['features']))
```

    the top 50 features are: 
    ['H_L0.01_weight', 'MI_dir_L0.01_weight', 'HH_L0.01_magnitude', 'MI_dir_L0.01_mean', 'H_L0.01_mean', 'HH_jit_L1_mean', 'H_L0.1_variance', 'MI_dir_L5_weight', 'H_L5_weight', 'H_L0.1_weight', 'H_L1_weight', 'MI_dir_L0.1_weight', 'MI_dir_L1_weight', 'H_L3_weight', 'MI_dir_L3_weight', 'H_L0.01_variance', 'MI_dir_L0.01_variance', 'HH_jit_L5_weight', 'HH_jit_L3_weight', 'MI_dir_L0.1_variance', 'HH_L5_weight', 'HH_jit_L0.1_mean', 'HH_L5_magnitude', 'HH_L3_weight', 'H_L1_mean', 'MI_dir_L1_mean', 'HH_L0.01_mean', 'HH_jit_L0.1_weight', 'HH_jit_L0.01_weight', 'HH_L0.1_weight', 'HH_L0.1_magnitude', 'HH_L1_magnitude', 'HH_L0.01_weight', 'HH_jit_L3_variance', 'H_L3_mean', 'MI_dir_L3_mean', 'HH_L1_weight', 'MI_dir_L5_mean', 'H_L5_mean', 'MI_dir_L0.1_mean', 'H_L0.1_mean', 'HH_jit_L1_weight', 'HH_jit_L5_variance', 'HH_jit_L1_variance', 'HH_L3_magnitude', 'HH_jit_L3_mean', 'HH_jit_L0.1_variance', 'HH_L0.1_mean', 'HH_L0.1_std', 'HH_L1_mean']
    
    the top 25 features are: 
    ['H_L0.01_weight', 'MI_dir_L0.01_weight', 'HH_L0.01_magnitude', 'MI_dir_L0.01_mean', 'H_L0.01_mean', 'HH_jit_L1_mean', 'H_L0.1_variance', 'MI_dir_L5_weight', 'H_L5_weight', 'H_L0.1_weight', 'H_L1_weight', 'MI_dir_L0.1_weight', 'MI_dir_L1_weight', 'H_L3_weight', 'MI_dir_L3_weight', 'H_L0.01_variance', 'MI_dir_L0.01_variance', 'HH_jit_L5_weight', 'HH_jit_L3_weight', 'MI_dir_L0.1_variance', 'HH_L5_weight', 'HH_jit_L0.1_mean', 'HH_L5_magnitude', 'HH_L3_weight', 'H_L1_mean']
    
    the top 10 features are: 
    ['H_L0.01_weight', 'MI_dir_L0.01_weight', 'HH_L0.01_magnitude', 'MI_dir_L0.01_mean', 'H_L0.01_mean', 'HH_jit_L1_mean', 'H_L0.1_variance', 'MI_dir_L5_weight', 'H_L5_weight', 'H_L0.1_weight']
    
    the top 5 features are: 
    ['H_L0.01_weight', 'MI_dir_L0.01_weight', 'HH_L0.01_magnitude', 'MI_dir_L0.01_mean', 'H_L0.01_mean']
    

#### <span style="color:#0b486b">2.2. Using different machine learning techniques for classification </span>
Extracting sub dataset for machine learning in each case


```python
X_T50 = X[ranked_df.head(n=50)['features']] #extract data of top 50 only
X_T25 = X[ranked_df.head(n=25)['features']] #extract data of top 25 only
X_T10 = X[ranked_df.head(n=10)['features']] #extract data of top 10 only
X_T5 = X[ranked_df.head(n=5)['features']] #extract data of top 5 only
```

Set up the lists to store values


```python
ls_T50 = []
ls_T25 = []
ls_T10 = []
ls_T5 = []
```

#### <span style="color:#0b486b">Apply KNeighborsClassifier() </span>


```python
kneig = KNeighborsClassifier()

# Apply to top 50 only
Scores50 = cross_val_score(kneig, X_T50, y, cv=10, scoring='accuracy')
ls_T50.append(str(round(np.mean(Scores50) * 100,2)) + "%")

#Apply to top 25 only
Scores25 = cross_val_score(kneig, X_T25, y, cv=10, scoring='accuracy')
ls_T25.append(str(round(np.mean(Scores25) * 100,2)) + "%")

#Apply to top 10 only
Scores10 = cross_val_score(kneig, X_T10, y, cv=10, scoring='accuracy')
ls_T10.append(str(round(np.mean(Scores10) * 100,2)) + "%")

#Apply to top 5 only
Scores5 = cross_val_score(kneig, X_T5, y, cv=10, scoring='accuracy')
ls_T5.append(str(round(np.mean(Scores5) * 100,2)) + "%")
```

#### <span style="color:#0b486b">Apply DecisionTreeClassifier() </span>


```python
tree =  DecisionTreeClassifier()

#Apply to top 50 only
Scores50 = cross_val_score(tree, X_T50, y, cv=10, scoring='accuracy')
ls_T50.append(str(round(np.mean(Scores50) * 100,2)) + "%")

#Apply to top 25 only
Scores25 = cross_val_score(tree, X_T25, y, cv=10, scoring='accuracy')
ls_T25.append(str(round(np.mean(Scores25) * 100,2)) + "%")

#Apply to top 10 only
Scores10 = cross_val_score(tree, X_T10, y, cv=10, scoring='accuracy')
ls_T10.append(str(round(np.mean(Scores10) * 100,2)) + "%")

#Apply to top 5 only
Scores5 = cross_val_score(tree, X_T5, y, cv=10, scoring='accuracy')
ls_T5.append(str(round(np.mean(Scores5) * 100,2)) + "%")
```

#### <span style="color:#0b486b">Apply GaussianNB() </span>


```python
gau = GaussianNB()

#Apply to top 50 only
Scores50 = cross_val_score(gau, X_T50, y, cv=10, scoring='accuracy')
ls_T50.append(str(round(np.mean(Scores50) * 100,2)) + "%")

#Apply to top 25 only
Scores25 = cross_val_score(gau, X_T25, y, cv=10, scoring='accuracy')
ls_T25.append(str(round(np.mean(Scores25) * 100,2)) + "%")

#Apply to top 10 only
Scores10 = cross_val_score(gau, X_T10, y, cv=10, scoring='accuracy')
ls_T10.append(str(round(np.mean(Scores10) * 100,2)) + "%")

#Apply to top 5 only
Scores5 = cross_val_score(gau, X_T5, y, cv=10, scoring='accuracy')
ls_T5.append(str(round(np.mean(Scores5) * 100,2)) + "%")
```

#### <span style="color:#0b486b">Apply RandomForestClassifier(n_estimators=10) </span>


```python
forest = RandomForestClassifier(n_estimators=10)

#Apply to top 50 only
Scores50 = cross_val_score(forest, X_T50, y, cv=10, scoring='accuracy')
ls_T50.append(str(round(np.mean(Scores50) * 100,2)) + "%")

#Apply to top 25 only
Scores25 = cross_val_score(forest, X_T25, y, cv=10, scoring='accuracy')
ls_T25.append(str(round(np.mean(Scores25) * 100,2)) + "%")

#Apply to top 10 only
Scores10 = cross_val_score(forest, X_T10, y, cv=10, scoring='accuracy')
ls_T10.append(str(round(np.mean(Scores10) * 100,2)) + "%")

#Apply to top 5 only
Scores5 = cross_val_score(forest, X_T5, y, cv=10, scoring='accuracy')
ls_T5.append(str(round(np.mean(Scores5) * 100,2)) + "%")
```

#### <span style="color:#0b486b">Apply AdaBoostClassifier()  </span>


```python
ada = AdaBoostClassifier() 

#Apply to top 50 only
Scores50 = cross_val_score(ada, X_T50, y, cv=10, scoring='accuracy')
ls_T50.append(str(round(np.mean(Scores50) * 100,2)) + "%")

#Apply to top 25 only
Scores25 = cross_val_score(ada, X_T25, y, cv=10, scoring='accuracy')
ls_T25.append(str(round(np.mean(Scores25) * 100,2)) + "%")

#Apply to top 10 only
Scores10 = cross_val_score(ada, X_T10, y, cv=10, scoring='accuracy')
ls_T10.append(str(round(np.mean(Scores10) * 100,2)) + "%")

#Apply to top 5 only
Scores5 = cross_val_score(ada, X_T5, y, cv=10, scoring='accuracy')
ls_T5.append(str(round(np.mean(Scores5) * 100,2)) + "%")
```

### <span style="color:#0b486b">3. Results and discussion </span>


```python
classifiers = ['KNeighbors', 'DecisionTree','GaussianNB', 'RandomForest', 'AdaBoost']

df_ML = pd.DataFrame()
df_ML['Classifiers'] = classifiers
df_ML['Accuracy top 50 features'] = ls_T50
df_ML['Accuracy top 25 features'] = ls_T25
df_ML['Accuracy top 10 features'] = ls_T10
df_ML['Accuracy top 5 features'] = ls_T5

df_ML
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
      <th>Classifiers</th>
      <th>Accuracy top 50 features</th>
      <th>Accuracy top 25 features</th>
      <th>Accuracy top 10 features</th>
      <th>Accuracy top 5 features</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KNeighbors</td>
      <td>98.92%</td>
      <td>98.66%</td>
      <td>99.3%</td>
      <td>99.88%</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DecisionTree</td>
      <td>99.84%</td>
      <td>99.82%</td>
      <td>99.86%</td>
      <td>99.88%</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GaussianNB</td>
      <td>53.92%</td>
      <td>53.6%</td>
      <td>54.28%</td>
      <td>99.88%</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RandomForest</td>
      <td>99.88%</td>
      <td>99.86%</td>
      <td>99.88%</td>
      <td>99.88%</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AdaBoost</td>
      <td>99.84%</td>
      <td>99.86%</td>
      <td>99.88%</td>
      <td>99.86%</td>
    </tr>
  </tbody>
</table>
</div>



From the results, it is very interesting that all classifier perform almost the best when only using 5 features. This can be explained that more data can create more noises (noisy data) that provide false or unneeded insights that will create false results in the end. However, from all the classifier, KNeighbors, Decision Tree, RandomForest and AdaBoost have extremely high results for all the cases. This also because that the malacious data have an extreme different set of data that make the classifiers easier in detecting them. All in all, based on the final results, RandomForest has the best result that is consistent through out the cases while the GaussianNB has the worst results. 

### <span style="color:#0b486b">4. References: </span>

[1]: Y. Mirsky, T. Doitshman, Y. Elovici & A. Shabtai 2018, 'Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection', in Network and Distributed System Security (NDSS) Symposium, San Diego, CA, USA.

[2]: Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, D. Breitenbacher, A. Shabtai, and Y. Elovici 'N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders', IEEE Pervasive Computing, Special Issue - Securing the IoT (July/Sep 2018).

[3]: Sze, Vivienne, et al. "Hardware for machine learning: Challenges and opportunities." 2017 IEEE Custom Integrated Circuits Conference (CICC). IEEE, 2017.
