# Propensity Score Matching Package for Causal Evaluation


```python

# python 代码
#原始包

# load packages
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()  # set the style
import matplotlib 
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *

#Data Prep
def data_prep(file_name):
    df_psmpy = pd.read_csv(file_name)
    print('\n',df_psmpy.info())
    print('sample:\n',df_psmpy.head())
    print('data type:\n',df_psmpy.is_treatment.value_counts())
    print('data describe:\n',df_psmpy.describe())
    
    #declare variables
    lines,cols=df_psmpy.shape
    group=int(np.ceil(lines/8000))
    #print('group_num is ',group)
    
    
    #【hash随机分组】
    df_psmpy['hash'] = df_psmpy.caseid.map(lambda x:hash(str(x)+'psm')%group)
    # print('sample:\n',df_psmpy.head())
    print('group num count :\n',df_psmpy.hash.value_counts())
    
    print('corr',df_psmpy.corr()[['is_treatment']])
    
    return df_psmpy


def psm_process(df_psmpy,i,exclude_array=[]): 
    print('this group is:',i)
    df_psmpy_hash = df_psmpy[df_psmpy['hash']==i]
    df_psmpy_hash.is_treatment.value_counts()
    # print('columns:' ,df_psmpy_hash.columns)
    
    #【Instantiate PsmPy Class】
    # exclude: ignore any covariates (columns) passed to the it during model fitting
    # indx - required parameter that references a unique ID number for each case
    exclude_array.append('hash') #分组字段不作为特征
    psm = PsmPy(df_psmpy_hash, treatment='is_treatment', indx='caseid',
                exclude= exclude_array)  # 'validwatchtime'
    
    #【Propensity Score】
    #There often exists a significant class imbalance in the data. This will be detected automatically. We account for this by setting balance=True when calling psm.logistic_ps().
    psm.logistic_ps(balance=True)
    print('group',i,'predicted_data \n',psm.predicted_data)
    
    sns.histplot(data=psm.predicted_data, x='propensity_score', hue='is_treatment',kde='true')
    plt.show()
    # multiple="dodge" 分组显示
    # kde='true',直方图内显示折线的核密度
    
    
    #【matching method1】
    # psm.knn_matched(matcher='propensity_score', replacement=False, caliper=None) #propensity_logit
    psm.knn_matched_12n(matcher='propensity_score', how_many=1)
    
    #【Graphical Outputs】
    # psm.plot_match(Title='Matching Result', Ylabel='# number of cases', Xlabel= 'propensity logit', names = ['treatment', 'control'])
    # plt.show()
    
    
    #【Plot the effect sizes of matching】
    print(psm.effect_size_plot()) 
    plt.show()
    print('group',i,'effect_size \n', psm.effect_size)
    
    
    #【show matched date】
    #print('\n group',i,'matched \n',psm.df_matched)
    #returns a dataframe of calculated propensity scores and propensity logits for all cases in the dataframe
    print('\n group',i,'matched \n',psm.matched_ids.nunique())
    
    
    #【merge match result】
    df_ps_match = psm.matched_ids.merge(psm.df_matched[['caseid', 'propensity_score', 'propensity_logit','is_treatment']],\
        how='inner', left_on='caseid', right_on='caseid').merge(psm.df_matched[['caseid', 'propensity_score', 'propensity_logit','is_treatment']], how='inner', left_on='largerclass_0group', right_on='caseid', suffixes=['_ori', '_mat'])
    
    df_matched1 = psm.matched_ids.merge(psm.df_matched[['caseid', 'propensity_score', 'propensity_logit','is_treatment']],how='inner', left_on='caseid', right_on='caseid')[['caseid','propensity_score', 'propensity_logit','is_treatment']]
    #print('match group1',df_matched1)
    df_matched2 = psm.matched_ids.merge(psm.df_matched[['caseid', 'propensity_score', 'propensity_logit','is_treatment']],how='inner', left_on='largerclass_0group', right_on='caseid')[['largerclass_0group','propensity_score'
                 ,'propensity_logit','is_treatment']].rename(columns = {'largerclass_0group':'caseid'})
    #print('match group2',df_matched2)
    
    
    #【concat two group】
    df_ps_match_res = pd.concat([df_matched1,df_matched2],axis = 0)

    sns.histplot(data=df_ps_match_res, x='propensity_score',hue='is_treatment')  
    plt.show()
    
    
    #【save result】
    df_ps_match_res['hash_group']=i
    df_ps_match_res.to_csv('PSM_result.csv',index=False)
println()

```
