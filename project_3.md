


### 目前进度：
* 模型构成 = 趋势（分段logistic） + 月/季度周期 + 假期因素 + *外部变量*  
* *三科总续报率拟合与异常判定* + *外部5个变量（主要渠道、城市等级，级别、年龄级别匹配、购课方式占比）*  
* *待确定：预测准确度与异常检测是否需要两套参数，如果用于异常检测的参数过于复杂，则可能失去检测意义*  
    * *过拟合：预测不准*  
    * *预测太准：不报异常*  
* *异常返回*：  
    * *异常返回输出excel*  
    * *异常返回格式修改*  

### 模型设置：
* **趋势项：**    
    * logistic growth   
    * 最低值和最高值的设定：历史最低值打5折/100%  
    * 关键时间点确定：英语s1 2020/9 + 自动选取的增长率改变节点
* **季节项：**  
    * 加入月/季度周期  
* **节假日项：**  
    * 手动输入：元旦/春节/国庆/五一/端午/清明/寒暑假
* **外部变量：**  
    * 主要渠道、城市等级，级别、年龄级别匹配、购课方式占比
* **城市类型分四类：** 一线/新一线/二线/三线以下  
* **模型评价指标：** 暂定RMSE（占比、续报率数值较小，所以暂时不用MAPE相关指标）

### 指标异常判断标准  
* 真实值是否在模型预期（置信区间）之外 或 真实值与趋势的偏离程度 
    * 调整置信区间大小的方法：  
        * 改变模型复杂度: 调整相关的先验概率参数。
        * 改变置信区间参数: 调整interval_width
        * 占比高的特征可根据权重设置较窄的区间
* 异常返回：
    * 异常期数/续报率真实值/续报率预测值/续报率置信区间/正负影响/占比同样/真实值总贡献/模型预测值总贡献/异常贡献
* 判断标准需用业务认知进行矫正  

### 后续优化方向  
* 根据异常贡献大小输出排名以获得对异常贡献最大的单维因素 ✔️ 
* 整期预测 + 外部5个维度（渠道、城市等级，级别、年龄级别匹配、购课方式等）——> 证明模型可信度高 ✔️
* 异常看板：输出excel  ✔️  
* 进入参数精修阶段
* 迁移到jupyterhub服务器，可先用24号开课日做测试


```python

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import seaborn as sns 
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
sns.set_palette(palette='deep')
sns_c = sns.color_palette(palette='deep')

#facebook Prophet
from prophet import Prophet
from prophet.plot import add_changepoints_to_plot,plot_plotly
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None) #展示所有列
import itertools
from itertools import islice
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 5000)
pd.set_option('expand_frame_repr', False)
pd.set_option('max_colwidth', 200)
pd.set_option("display.colheader_justify","right")

from pyspark.sql import *
spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.executor.instances", "20") \
    .config("spark.executor.cores", "1") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .enableHiveSupport() \
    .getOrCreate()
spark

# 城市类型
city = spark.sql("""

select subject, period_type, total_count, 
		case when city_type = '一线城市' then '一线城市'
        	 when city_type = '二线城市' then '二线城市'
             when city_type = '新一线' then '新一线'
			 else '三线城市及以下' end as city_type,
			   count(distinct userid) as user_num,
			   count(distinct if(a.if_w2d6_xu  = 1, a.userid, null)) as xubao_num
		from
		(
			select 
    			  userid, subject, semesterid, period_type, stage_name, wk_dt, is_season, buy_season_dt, city_type, age_buy_lesson_type,
            is_age_match, teacher_city, order_label, buy_way, if(is_season = 1 and buy_season_dt <= date_sub(wk_dt,1), 1, 0) as if_w2d6_xu,
            user_from, ad_channel,
            count(userid) over (partition by period_type) as total_count
    		from eng.dm_eng_try_double_user_quality_detail
        	where dt = date_sub(current_date,1)
          	and subject in (1, 2, 3) --三科
          	and start_class_dt between '2020-10-04' and '2021-10-25' 
    ) a
    group by subject, period_type, total_count, 
    		case when city_type = '一线城市' then '一线城市'
        	 when city_type = '二线城市' then '二线城市'
             when city_type = '新一线' then '新一线'
			 else '三线城市及以下' end 

""").toPandas()

# 级别
stage = spark.sql("""

select subject, period_type, total_count, stage_name,
             count(distinct userid) as user_num,
             count(distinct if(a.if_w2d6_xu  = 1, a.userid, null)) as xubao_num
        from
        (
          select 
                userid, subject, semesterid, period_type, stage_name, wk_dt, is_season, buy_season_dt, city_type, age_buy_lesson_type,
                is_age_match, teacher_city, order_label, buy_way, if(is_season = 1 and buy_season_dt <= date_sub(wk_dt,1), 1, 0) as if_w2d6_xu,
                user_from, ad_channel,
                count(userid) over(partition by period_type) as total_count
            from eng.dm_eng_try_double_user_quality_detail
              where dt = date_sub(current_date,1)
                and subject in (1, 2, 3) --三科
                and start_class_dt between '2020-03-30' and '2021-06-07'   
        ) a
        group by subject, period_type, total_count, stage_name

""").toPandas()

# 年龄级别匹配
agematch = spark.sql("""

select subject, period_type, total_count, is_age_match,
             count(distinct userid) as user_num,
             count(distinct if(a.if_w2d6_xu  = 1, a.userid, null)) as xubao_num
        from
        (
          select 
                userid, subject, semesterid, period_type, stage_name, wk_dt, is_season, buy_season_dt, city_type, age_buy_lesson_type,
                is_age_match, teacher_city, order_label, buy_way, if(is_season = 1 and buy_season_dt <= date_sub(wk_dt,1), 1, 0) as if_w2d6_xu,
                user_from, ad_channel,
                count(userid) over(partition by period_type) as total_count
            from eng.dm_eng_try_double_user_quality_detail
              where dt = date_sub(current_date,1)
                and subject in (1, 2, 3) --三科
                and start_class_dt between '2020-03-30' and '2021-06-07'   
        ) a
        group by subject, period_type, total_count, is_age_match

""").toPandas()

# 购课方式
buyway = spark.sql("""

select a.subject, a.period_type, a.total_count, a.buy_way,
             count(distinct userid) as user_num,
             count(distinct if(a.if_w2d6_xu  = 1, a.userid, null)) as xubao_num
        from
        (
          select 
                userid, subject, semesterid, period_type, stage_name, wk_dt, is_season, buy_season_dt, city_type, age_buy_lesson_type,
                is_age_match, teacher_city, order_label, buy_way, if(is_season = 1 and buy_season_dt <= date_sub(wk_dt,1), 1, 0) as if_w2d6_xu,
                user_from, ad_channel,
                count(userid) over(partition by period_type) as total_count
            from eng.dm_eng_try_double_user_quality_detail
              where dt = date_sub(current_date,1)
                and subject in (1, 2, 3) --三科
                and start_class_dt between '2020-03-30' and '2021-06-07'   
        ) a
        group by a.subject, a.period_type, a.total_count, a.buy_way
""").toPandas()

# 渠道
userfrom = spark.sql("""

select subject, period_type, total_count, 
        case
            when user_from='外部推广' and ad_channel='朋友圈' then '01-外部推广-朋友圈'
            when user_from='外部推广' and ad_channel='抖音' then '02-外部推广-抖音'
            when user_from='外部推广' and ad_channel='广点通' then '03-外部推广-广点通'
            when user_from='外部推广' and ad_channel not in ('广点通','朋友圈','抖音') then '04-外部推广-其他'
            when user_from='营销增长' then '05-营销增长'
            when user_from='扩科' and order_label='已购任意一科长期课' then '06-扩科-系统课'
            when user_from='扩科' and order_label='已购任意一科双周课/首月课' then '07-扩科-双周课'
            when user_from='扩科' and order_label not in ('已购任意一科双周课/首月课','已购任意一科长期课') then '08-扩科-其他'
            when user_from='转介绍' then '09-转介绍'
            when user_from in ('导流课转化','试听课转化') then '10-导流课/试听课转化'
            else '11-其他' 
        end as user_from,
             count(distinct userid) as user_num,
             count(distinct if(a.if_w2d6_xu  = 1, a.userid, null)) as xubao_num
        from
        (
          select 
                userid, subject, semesterid, period_type, stage_name, wk_dt, is_season, buy_season_dt, city_type, age_buy_lesson_type,
                is_age_match, teacher_city, order_label, buy_way, if(is_season = 1 and buy_season_dt <= date_sub(wk_dt,1), 1, 0) as if_w2d6_xu,
                user_from, ad_channel,
                count(userid) over(partition by period_type) as total_count
            from eng.dm_eng_try_double_user_quality_detail
              where dt = date_sub(current_date,1)
                and subject in (1, 2, 3) --三科
                and start_class_dt between '2020-03-30' and '2021-06-07'   
        ) a
        group by subject, 
                 period_type, 
                 total_count, 
              case when user_from='外部推广' and ad_channel='朋友圈' then '01-外部推广-朋友圈'
                    when user_from='外部推广' and ad_channel='抖音' then '02-外部推广-抖音'
                    when user_from='外部推广' and ad_channel='广点通' then '03-外部推广-广点通'
                    when user_from='外部推广' and ad_channel not in ('广点通','朋友圈','抖音') then '04-外部推广-其他'
                    when user_from='营销增长' then '05-营销增长'
                    when user_from='扩科' and order_label='已购任意一科长期课' then '06-扩科-系统课'
                    when user_from='扩科' and order_label='已购任意一科双周课/首月课' then '07-扩科-双周课'
                    when user_from='扩科' and order_label not in ('已购任意一科双周课/首月课','已购任意一科长期课') then '08-扩科-其他'
                    when user_from='转介绍' then '09-转介绍'
                    when user_from in ('导流课转化','试听课转化') then '10-导流课/试听课转化'
                    else '11-其他' end
""").toPandas()

# 是否乡镇
xiangzhen = spark.sql("""

select a.subject, a.period_type, a.total_count, b.if_xiangzhen,
             count(distinct a.userid) as user_num,
             count(distinct if(a.if_w2d6_xu  = 1, a.userid, null)) as xubao_num
        from
        (
          select 
                userid, subject, semesterid, period_type, stage_name, wk_dt, is_season, buy_season_dt, city_type, age_buy_lesson_type,
                is_age_match, teacher_city, order_label, buy_way, if(is_season = 1 and buy_season_dt <= date_sub(wk_dt,1), 1, 0) as if_w2d6_xu,
                user_from, ad_channel,
                count(userid) over(partition by period_type) as total_count
            from eng.dm_eng_try_double_user_quality_detail
              where dt = date_sub(current_date,1)
                and subject in (1, 2, 3) --三科
                and start_class_dt between '2020-03-30' and '2021-06-07'  
        ) a
        join 
        (
          select
            userid,
            subject,
            city,
            city_type,
            address,
            if( address rlike '村|镇|乡','乡镇','非乡镇') as if_xiangzhen
        from eng.dw_eng_order_user_address_da
        where type = '1'
        ) b
        on a.userid = b.userid and a.subject = b.subject
        group by a.subject, a.period_type, a.total_count, b.if_xiangzhen
""").toPandas()

# 总续报率
total_xubao = spark.sql("""

select subject, period_type, total_count,
             count(distinct userid) as user_num,
             count(distinct if(a.if_w2d7_xu  = 1, a.userid, null)) as xubao_num
        from
        (
          select 
                userid, subject, semesterid, period_type, stage_name, wk_dt, is_season, buy_season_dt, city_type, age_buy_lesson_type,
                is_age_match, teacher_city, order_label, buy_way, if(is_season = 1 and buy_season_dt <= wk_dt, 1, 0) as if_w2d7_xu,
                user_from, ad_channel,
                count() over(partition by period_type) as total_count
            from eng.dm_eng_try_double_user_quality_detail
              where dt = date_sub(current_date,1)
                and subject in (1, 2, 3) --三科
                and start_class_dt between '2020-03-30' and '2021-06-07'  
        ) a
        group by subject, period_type, total_count
""").toPandas()
# 老师城市
teacher_city = spark.sql("""

select subject, period_type, total_count, teacher_city, stage_name,
			   count(distinct userid) as user_num,
			   count(distinct if(a.if_w2d7_xu  = 1, a.userid, null)) as xubao_num
		from
		(
			select 
    			  userid, subject, semesterid, period_type, stage_name, wk_dt, is_season, buy_season_dt, city_type, age_buy_lesson_type,
            is_age_match, teacher_city, order_label, buy_way, if(is_season = 1 and buy_season_dt <= wk_dt, 1, 0) as if_w2d7_xu,
            user_from, ad_channel,
            count(userid) over (partition by period_type) as total_count
    		from eng.dm_eng_try_double_user_quality_detail
        	where dt = date_sub(current_date,1)
          	and subject in (1, 2, 3) --三科
          	and start_class_dt between '2020-03-30' and '2021-06-07' 
    ) a
    group by subject, period_type, total_count, teacher_city, stage_name

""").toPandas()
userfrom[(userfrom['subject'] == '英语')&(userfrom['user_from'] == '01-外部推广-朋友圈')].info()
city[(city['subject'] == '英语')&(city['city_type'] == '一线城市')].info()
# 更改学科特征值
city['subject'] = city['subject'].replace({'1':'英语','2':'思维', '3':'语文'})
stage['subject'] = stage['subject'].replace({'1':'英语','2':'思维', '3':'语文'})
agematch['subject'] = agematch['subject'].replace({'1':'英语','2':'思维', '3':'语文'})
buyway['subject'] = buyway['subject'].replace({'1':'英语','2':'思维', '3':'语文'})
userfrom['subject'] = userfrom['subject'].replace({'1':'英语','2':'思维', '3':'语文'})
xiangzhen['subject'] = xiangzhen['subject'].replace({'1':'英语','2':'思维', '3':'语文'})
teacher_city['subject'] = teacher_city['subject'].replace({'1':'英语','2':'思维', '3':'语文'})
teacher_city['subject'] = teacher_city['subject'].replace({'1':'英语','2':'思维', '3':'语文'})

# 购课方式 保证时间序列连续
buyway = buyway.drop(buyway[(buyway['period_type'] == '11-2020/04/13') & (buyway['subject'] == '英语') & (buyway['buy_way'] == '联报')].index.tolist())

# 渠道来源
insertRow = pd.DataFrame([['英语','21-2020/06/22', 70433, '03-外部推广-广点通', 1, 1]],columns = ['subject','period_type','total_count','user_from','user_num', 'xubao_num'])
userfrom = pd.concat([userfrom, insertRow], ignore_index = True)

userfrom[(userfrom['subject'] == '英语')&(userfrom['user_from'] == '03-外部推广-广点通')&(userfrom['period_type'] == '21-2020/06/22')]

# 级别
stage = stage.drop(stage[stage['stage_name']=='S4'].index.tolist())

# 级别年龄匹配
agematch['is_age_match'] = agematch['is_age_match'].replace({'1':'是', '0':'否'})
# 输出某特征续报率时间序列
def data_xubaoprep(data):
    d = {}
    for i in data[data.columns[0]].unique():
        d[i]={}
        for j in data[data.columns[3]].unique():
            ts = data[(data[data.columns[0]] == i) & (data[data.columns[3]] == j)]
            ts['续报率'] = ts['xubao_num']/ts['user_num'] # 续报率计算
            ts['period_type'] = ts.apply(lambda x: x['period_type'][-10:], axis=1) # 日期
            ts['period_type'] = pd.to_datetime(ts['period_type'])
            ts = ts.sort_values(by = 'period_type').rename(columns = {"period_type":"ds","续报率":"y"}).reset_index()
            d[i][j] = ts
    return d

# 学科+老师城市+级别
def data_conversion(data, date):
    d = {}
    for i in data[data.columns[0]].unique():
        d[i] = {}
        for j in data[data.columns[3]].unique():
            d[i][j]={}
            for n in data[data.columns[4]].unique():
                ts = data[(data[data.columns[0]] == i) & (data[data.columns[3]] == j) & (data[data.columns[4]] == n)]
                if ts.empty:
                    pass
                else:
                    ts['续报率'] = ts['xubao_num']/ts['user_num'] # 续报率计算
                    ts['period_type'] = ts.apply(lambda x: x['period_type'][-10:], axis=1) # 日期
                    ts['period_type'] = pd.to_datetime(ts['period_type'])
                    ts = ts.sort_values(by = 'period_type').rename(columns = {"period_type":"ds","续报率":"y"}).reset_index()
                    #填补缺失值
                    date_index = pd.date_range(ts['ds'].min(),date, freq = '7D')
                    ts = ts.set_index('ds').reindex(date_index).fillna(0).reset_index().rename(columns = {"level_0":"ds"})
                    ts['subject'] = ts['subject'].replace(0, i)
                    ts['teacher_city'] = ts['teacher_city'].replace(0, j)
                    ts.loc[ts['y']==0,'y'] = None
                    d[i][j][n] = ts[['ds', 'y']]
    return d

for subject_name, cities in xubao_teachercity.items():
    for city_name, stages in cities.items():
            for stage_name, values in stages.items():
                print(subject_name +'+'+ city_name+'+'+stage_name)

# 输出某特征占比时间序列
def data_zhanbiprep(data):
    d = {}
    for i in data[data.columns[0]].unique():
        d[i] = {}
        for j in data[data.columns[3]].unique():
            ts = data[(data[data.columns[0]] == i) & (data[data.columns[3]] == j)]
            ts['占比'] = ts['user_num']/ts['total_count'] # 占比计算
            ts['period_type'] = ts.apply(lambda x: x['period_type'][-10:], axis=1) # 日期
            ts['period_type'] = pd.to_datetime(ts['period_type'])
            ts = ts.sort_values(by = 'period_type').rename(columns = {"period_type":"ds","占比":"y"}).reset_index()
            d[i][j] = ts
    return d

```
## 总续报率预测

### 数据准备

**外部变量**  
* 级别：
    * 英语：s2/s3  
    * 思维：s2/s3  
    * 语文：s2/s3  
    
* 购课方式：单科
* 城市类型：一线/新一线/二线/三线及以下
* 年纪匹配：是
* 渠道：朋友圈/抖音/营销增长/扩科-系统课/扩科-双周课/转介绍
```python

# total_xubao = pd.read_csv('三科总续报率.csv')
total_xubao['subject'] = total_xubao['subject'].replace({'1':'英语','2':'思维', '3':'语文'})

len(total_xubao['period_type'].unique())

def xubaoprep(data):
    d = {}
    for i in data[data.columns[0]].unique():
        ts = data[(data[data.columns[0]] == i)]
        ts['续报率'] = ts['xubao_num']/ts['user_num'] # 续报率计算
        ts['period_type'] = ts.apply(lambda x: x['period_type'][-10:], axis=1) # 日期
        ts['period_type'] = pd.to_datetime(ts['period_type'])
        ts = ts.sort_values(by = 'period_type')
        ts = ts.rename(columns = {"period_type":"ds","续报率":"y"})
        ts = ts.reset_index()
        d[i] = ts
    return d

# 输出三科续报率
xubao_rate = xubaoprep(total_xubao)

q={}
for subjects in xubao_rate.keys():
    q[subjects] = xubao_rate[subjects][['ds', 'y']]
    # 级别
    q[subjects]['S2占比'] = zhanbi_stage[subjects]['S2']['y']
    q[subjects]['S3占比'] = zhanbi_stage[subjects]['S3']['y']
    # 购课方式
    q[subjects]['单科占比'] = zhanbi_buyway[subjects]['单科']['y']
    # 级别年龄匹配
    q[subjects]['级别年龄匹配'] = zhanbi_agematch[subjects]['是']['y']
    # 城市类型
    q[subjects]['一线城市占比'] = zhanbi_city[subjects]['一线城市']['y']
    q[subjects]['二线城市占比'] = zhanbi_city[subjects]['二线城市']['y']
    q[subjects]['新一线占比'] = zhanbi_city[subjects]['新一线']['y']
    q[subjects]['三线城市及以下占比'] = zhanbi_city[subjects]['三线城市及以下']['y']
    # 渠道
    q[subjects]['朋友圈占比'] = zhanbi_userfrom[subjects]['01-外部推广-朋友圈']['y']
    q[subjects]['抖音占比'] = zhanbi_userfrom[subjects]['02-外部推广-抖音']['y']
    q[subjects]['扩科系统课占比'] = zhanbi_userfrom[subjects]['06-扩科-系统课']['y']
    q[subjects]['扩科双周课占比'] = zhanbi_userfrom[subjects]['07-扩科-双周课']['y']
    q[subjects]['转介绍占比'] = zhanbi_userfrom[subjects]['09-转介绍']['y']

# 外部变量
regressor = ['S2占比', 'S3占比','级别年龄匹配','一线城市占比', '二线城市占比','新一线占比','三线城市及以下占比','单科占比',
             '朋友圈占比', '抖音占比','扩科系统课占比', '扩科双周课占比', '转介绍占比']

### 三科续报率预测与异常检测

---
##### sample check

# 调参
param_total, result_total = param_selection(q['英语'], '2021-06-07', 'logistic', holidays, '英语', regressor)

# 最佳参数
param_total, result_total

params_s = {'英语':{'changepoint_prior_scale': 2.0, 'season_prior_scale': 0.01, 'holidays_prior_scale': 0.1, 'regressor_prior_scale': 10},
            '思维': {'changepoint_prior_scale': 2.0, 'season_prior_scale': 0.01, 'holidays_prior_scale': 1.0, 'regressor_prior_scale': 0.01},
            '语文':{'changepoint_prior_scale':1, 'holidays_prior_scale': 10, 'prior_scale': 10}}

future_xubao_s, plot1_s, plot2_s, m_s = build_prophet(q['英语'], '2021-06-07', 'logistic', params_s['英语']['changepoint_prior_scale'], params_s['英语']['season_prior_scale'], params_s['英语']['holidays_prior_scale'], params_s['英语']['regressor_prior_scale'], holidays, '英语', regressor)

# 输出预测
prediction_nextweek = future_xubao_s[-1:]['yhat']

outlier_detection_total(future_xubao_s, 4, q['英语'], '英语')

future_xubao_s.tail(5)

---

xubao_rate

# 输出三科总续报率预测
future_xubao = {}
plot1_t= {}
plot2_t = {}
m_t = {}
params_t = {}

for subjects in xubao_rate.keys():
    params_t[subjects] = param_selection(q[subjects], '2021-05-24', 'logistic', holidays, subjects, regressor)
    future_xubao[subjects], plot1_t[subjects], plot2_t[subjects], m_t[subjects] = build_prophet(q[subjects], '2021-05-24', 'logistic', params_t[subjects]['changepoint_prior_scale'], params_t[subjects]['season_prior_scale'], params_t[subjects]['holidays_prior_scale'],params_t[subjects]['regressor_prior_scale'], holidays, subjects, regressor)
    outlier_detection_total(future_xubao[subjects], 4, q[subjects], subjects)

# 异常诊断
for subjects in xubao_rate.keys():
    outlier_detection_total(future_xubao[subjects], 4, q[subjects], subjects)

*异常诊断需要基于模型历史趋势的拟合来判断异常*

# 最近五期拟合预测详情
future_xubao['语文'].tail() 
future_xubao['英语'].tail() 
future_xubao['思维'].tail() 

---

## 各特征单维时间序列预测   
### 数据准备

# 原始数据画图
def plot_orig(data, title, ylabel):
    plt.rcParams['figure.figsize'] = [12, 7]
    fig, ax = plt.subplots()
    sns.lineplot(x='ds', y='y', label='y', data=data, ax=ax)
    ax.legend(loc='upper left')
    ax.set_ylim([0, 0.6])
    ax.locator_params("y", nbins = 25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set(title=title, xlabel='date', ylabel=ylabel)

# 所有特征组合
for subject, cities in xubao_userfrom.items():
    for key in cities:
        print(subject +'+'+ key)

# 输出城市类型续报率原始数据
for subject, cities in xubao_teachercity.items():
    for key in cities:
        plot_orig(xubao_teachercity[subject][key][['ds', 'y']], subject +'+'+ key,'续报率')

for subject_name, cities in xubao_teachercity.items():
    for city_name, stages in cities.items():
            for stage_name, values in stages.items():
                plot_orig(xubao_teachercity[subject_name][city_name][stage_name][['ds', 'y']], subject_name +'+'+ city_name+'+'+stage_name,'renewal rate')

# 输出级别续报原始数据
for subject, stages in xubao_stage.items():
    for key in stages:
        plot_orig(xubao_stage[subject][key][['ds', 'y']], subject +'+'+ key,'续报率')

# 输出级别占比原始数据
for subject, stages in xubao_stage.items():
    for key in stages:
        plot_orig(zhanbi_stage[subject][key][['ds', 'y']], subject +'+'+ key,'级别占比')

### 假期因素

# 假期因素
yuandan = pd.DataFrame({
  'holiday': '元旦',
  'ds': pd.to_datetime(['2020-12-21', '2020-12-28']), 'lower_window': -2, 'upper_window': 2,})
chunjie = pd.DataFrame({
  'holiday': '春节',
  'ds': pd.to_datetime(['2021-02-01', '2021-02-22']),
  'lower_window': -2, 'upper_window': 2,})
guoqing = pd.DataFrame({
  'holiday': '国庆',
  'ds': pd.to_datetime(['2020-09-21', '2020-10-12']),
  'lower_window': -2, 'upper_window': 2,})
wuyi = pd.DataFrame({
  'holiday': '五一',
  'ds': pd.to_datetime(['2020-04-20', '2020-04-27', '2020-05-04', '2021-04-25']),
  'lower_window': -2, 'upper_window': 2,})
qingming = pd.DataFrame({
  'holiday': '清明',
  'ds': pd.to_datetime(['2020-03-30', '2020-04-06', '2021-03-29','2021-04-05']),
  'lower_window': -2, 'upper_window': 2,})
duanwu = pd.DataFrame({
  'holiday': '端午',
  'ds': pd.to_datetime(['2020-06-15', '2020-06-22']),
  'lower_window': -2, 'upper_window': 2,})
hanshujia = pd.DataFrame({
  'holiday': '寒暑假',
  'ds': pd.to_datetime(['2020-06-29', '2020-07-06', '2020-07-13', '2020-07-20', '2020-07-27', '2020-08-03', '2020-08-10'
                       , '2020-08-17', '2020-08-24', '2020-08-31', '2021-01-11', '2021-01-18', '2021-01-25'
                       , '2021-02-01', '2021-02-22']),
  'lower_window': -3, 'upper_window': 3,})
holidays = pd.concat((yuandan, chunjie, guoqing, wuyi, qingming, duanwu, hanshujia))

### Prophet建模

# prophet建模
def build_prophet(data, date, trend, changepoint_prior_scale, season_prior_scale, holidays_prior_scale, regressor_prior_scale, holidays, feature, extra_features):
    
    # 划分训练集
    threshold_date = pd.to_datetime(date)
    mask = data['ds'] < threshold_date
    train = data[mask] 
    test = data[~mask]
    train['cap'] = max(data['y'])
    train['floor'] = min(data['y'])*0.5 
    # prophet模型拟合
    m = Prophet(growth = trend, yearly_seasonality = False, weekly_seasonality = False, daily_seasonality = False,
                changepoint_range = 0.95, changepoint_prior_scale = changepoint_prior_scale, holidays = holidays,
               interval_width = 0.8, holidays_prior_scale = holidays_prior_scale) 
    m.add_seasonality(name = 'monthly', period = 30.5, fourier_order = 5, prior_scale = season_prior_scale) 
    m.add_seasonality(name = 'quartly', period = 91.25, fourier_order = 8, prior_scale = season_prior_scale)
    
    if extra_features:
        for features in extra_features:
            m.add_regressor(features, standardize = 'auto', prior_scale = regressor_prior_scale)
    else:
        pass
    m.fit(train) 
    # 输出预测
    future = m.make_future_dataframe(periods = test.shape[0], freq = 'W')
    future['cap'] = max(data['y'])
    future['floor'] = min(data['y'])*0.5
    if extra_features:
        for keys in extra_features:
            future[keys] = data[keys]
    else:
        pass
    forecast = m.predict(future)
    # 画图
    fig, ax = plt.subplots()
    ax.set_ylim([0, 1.0])
    plot_1 = m.plot(forecast, ax = ax)
    ax.set_title(feature, fontsize = 14)
    a = add_changepoints_to_plot(plot_1.gca(), m, forecast) 
    plot_2 = m.plot_components(forecast) # 模型解构
    
    return forecast, plot_1, plot_2, m

def outlier_detection_total(a, prediction_size, xubao_data, feature):
    a['y_true'] = list(xubao_data['y'])
    l = a[-prediction_size:]
    index_xubao = list(np.where((l['y_true'] <= l["yhat_lower"])|
                     (l['y_true'] >= l["yhat_upper"]), True, False))
    
    outlier_xubao = l[index_xubao][['ds', 'yhat', 'y_true', 'yhat_lower', 'yhat_upper']]
    if len(outlier_xubao) == 0:
        print(feature, '总续报率无异常')   
    else:
        outlier_xubao['正负影响'] = outlier_xubao.apply(lambda x: '-' if x['y_true'] < x['yhat_lower'] else '+', axis =1)
        print(feature, '总续报率有异常值得关注:')
        print(outlier_xubao)

# 异常检测
def abnormal_detection(a, b, prediction_size, xubao_data, zhanbi_data, feature1, feature2):
    a['y_true'] = list(xubao_data['y'])
    b['y_true'] = list(zhanbi_data['y'])
    c = pd.DataFrame({'ds': a['ds'], '总贡献(续报率*占比)': a['y_true']*b['y_true']})
    c['模型总贡献(续报率*占比)'] = a['yhat']*b['yhat']
    c['异常贡献'] = c['总贡献(续报率*占比)'] - c['模型总贡献(续报率*占比)']
    l = a[-prediction_size:]
    m = b[-prediction_size:]
    n = c[-prediction_size:]
    
    l = l.rename(columns={'ds': '日期','y_true': '续报真实值','yhat': '续报预测值', 'yhat_lower': '续报预测下界', 'yhat_upper': '续报预测上界'})
    m = m.rename(columns={'ds': '日期','y_true': '占比真实值','yhat': '占比预测值', 'yhat_lower': '占比预测下界', 'yhat_upper': '占比预测上界'})
    n = n.rename(columns={'ds': '日期'})
                                                    
   
    l['续报正负影响'] = l.apply(lambda x: '-' if x['续报真实值'] < x['续报预测下界'] 
                                                    else ('无影响' if (x['续报真实值'] < x['续报预测上界'])&(x['续报真实值'] > x['续报预测下界']) else '+'), axis =1)

    m['占比正负影响'] = m.apply(lambda x: '-' if x['占比真实值'] < x['占比预测下界'] 
                                                      else ('无影响' if (x['占比真实值'] < x['占比预测上界'])&(x['占比真实值'] > x['占比预测下界']) else '+'), axis =1)
    #print(outlier_zhanbi, file = t)
    r = pd.merge(l, m, on = '日期', how='outer')
    o = pd.merge(r, n, on='日期', how='inner')
    o['学科']=feature1
    o['特征']=feature2
    columns = ['日期','学科', '特征','异常贡献', '续报真实值', '占比真实值','续报预测值','续报预测下界','续报预测上界','占比预测值', '占比预测下界', '占比预测上界', '续报正负影响','占比正负影响']
    return o[columns]

### sample check  
##### 老师城市  
2+chongqing+S1

# 续报率 example1： 2+chongqing+S1
forecast3, plot_1, plot_2, m1 = build_prophet(xubao_teachercity['2']['chongqing']['S1'][['ds', 'y']], 
                                         '2021-05-31', 'logistic', 1, 0.01, 1, 0.01, holidays, '2+chongqing+S1',[])

forecast3.tail() # 预测值
xubao_teachercity['2']['chongqing']['S1'][-1:] #真实值

forecast4, plot_2, plot_3, m2 = build_prophet(xubao_teachercity['2']['chengdu']['S3'][['ds', 'y']], 
                                         '2021-05-31', 'logistic', 2, 0.01, 1, 0.01, holidays, '2+tianjin+S3',[])

forecast4.tail() # 预测值
xubao_teachercity['2']['chengdu']['S3'][-5:] #真实值

# 占比 example2
forecast5, plot3, plot3, m2 = build_prophet(zhanbi_city['英语']['新一线'][['ds', 'y']], 
                                         '2021-05-17', 'logistic', 0.5, 0.01, 0.01, 1, holidays,'英语+新一线',[])

abnormal_detection(forecast3, forecast3, 7, xubao_teachercity['2']['chongqing']['S1'][['ds', 'y']], xubao_teachercity['2']['chongqing']['S1'][['ds', 'y']], '英语', '新一线')


```
**异常解释**
1. 异常贡献的参照：异常于模型预测值
2. 判断预测期的续报率和占比是否异常
3. 如有异常，判断预测期异常贡献大小
4. 异常贡献大的对总续报率的异常贡献大
5. 根据异常贡献排序归因出异常原因

```python
# 选取RMSE

##### 级别

# 续报率 example3
forecast7, plot7, plot8, m7 = build_prophet(xubao_stage['英语']['S2'][['ds', 'y']], 
                                         '2021-05-24', 'logistic', 0.01, 0.01, 0.1, 0.01, holidays,'英语+新一线',[])

forecast9, plot9, plot10, m9 = build_prophet(zhanbi_stage['英语']['S2'][['ds', 'y']], 
                                         '2021-05-24', 'logistic', 0.01, 0.01, 0.1, 0.01, holidays,'英语+新一线',[])

abnormal_detection(forecast7, forecast9, 9, xubao_stage['英语']['S2'][['ds', 'y']], zhanbi_stage['英语']['S2'][['ds', 'y']], '英语', 'S2')

### 输出预测

def prediction_output(xubao_data, holidays, regressor):
    prediction_xubao = {}
    prediction_zhanbi = {}
    plot1_xubao = {}
    plot1_zhanbi = {}
    plot2_xubao = {}
    plot2_zhanbi = {}
    params = {}
    results = {}
    m = {}
    data0 = pd.DataFrame()
    for subject, features in xubao_data.items():
        for value in features:
            params[subject +'+'+ value], results[subject +'+'+ value] = param_selection(xubao_teachercity[subject][value][['ds', 'y']], '2021-05-24', 'logistic', holidays = holidays, subject +'+'+ value, [])
            prediction_xubao[subject +'+'+ value], plot1_xubao[subject +'+'+ value], plot2_xubao[subject +'+'+ value], m[subject +'+'+ value] = build_prophet(xubao_teachercity[subject][value][['ds', 'y']],'2021-05-24', 'logistic', params[subject +'+'+ value]['changepoint_prior_scale'], params[subject +'+'+ value]['season_prior_scale'], params[subject +'+'+ value]['holidays_prior_scale'],params[subject +'+'+ value]['regressor_prior_scale'], holidays, subjects,regressor) # fit model
            data0 = pd.concat([data0, prediction_xubao[subject +'+'+ value][-1:]])
    return data0['yhat']

## Hyperparameter Tuning 
**目的：** 提高模型泛化能力  
**Trend:**
* linear or logistic  
* changepoint✔️  
     * automatically or mannually  
     * changepoint_prior_scale:  

**Holiday:**  
* 元旦国庆春节五一清明端午 
* holidays_prior_scale✔️

**Season:**
* 月/季度周期  
* prior_scale  

**Uncertainty:**  
* trend  
* seasonalty  

**方法：交叉验证**

help(cross_validation)

# 使用RMSE进行交叉验证 输出最佳参数组合
def param_selection(data, date, trend, holidays, feature, extra_features):
    param_grid = {  
    'changepoint_prior_scale': [0.5, 1.0, 2.0, 4.0],
    'season_prior_scale': [0.01],
    'holidays_prior_scale': [0.01, 0.1, 1.0, 10],
    'regressor_prior_scale': [0.01, 0.1, 1.0, 10],}
    
    all_params = [dict(zip(param_grid.keys(), i)) for i in itertools.product(*param_grid.values())] # 笛卡尔积所有组合
    rmses = [] # 存储rmses
    # 网格搜索
    for params in all_params:
        prediction, plot_1, plot_2, m = build_prophet(data = data, date = date , trend = trend, **params, holidays=holidays, feature=feature, extra_features=extra_features)
        cutoffs = pd.to_datetime(['2021-04-12','2021-04-19', '2021-05-03','2021-05-10', '2021-05-17','2021-05-24'])
        ts_cv = cross_validation(m, initial = '23 W', cutoffs = cutoffs, horizon = '1 W', parallel = "processes")
        ts_p = performance_metrics(ts_cv, rolling_window = 1)
        rmses.append(ts_p['rmse'].values[0])
    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    best_params = all_params[np.argmin(rmses)]
    return best_params, tuning_results

param = {'英语+一线城市':{'changepoint_prior_scale': 10.0, 'holidays_prior_scale': 0.001}}
param['英语+一线城市']['changepoint_prior_scale']

param_new,result_new = param_selection(xubao_stage['英语']['S2'][['ds', 'y']], '2021-05-24', 'logistic', holidays, '英语+S2',[])

param_new,result_new

best_params = {}
tuning_result = {}
for subjects in xubao_rate.keys():
    best_params[subjects],tuning_result[subjects] = cross_validat(q[subjects])

best_params

#输出 学科和城市类型 参数最佳组合
for subject, cities in xubao_city.items():
    for key in cities:
            cross_validat(xubao_city[subject][key][['ds', 'y']])

## 异常期数输出面板

# 备用code 1
params_xubao = {}
params_zhanbi = {}
for subject, features in xubao_data.items():
    for value in features:
    params_xubao[subject +'+'+ key] = cross_validat(xubao_city[subject][key][['ds', 'y']]) # select best params
    params_zhanbi[subject +'+'+ key] = cross_validat(zhanbi_city[subject][key][['ds', 'y']])

#备用code 2
, cv_xubao[subject +'+'+ value], p_xubao[subject +'+'+ value], plot3_xubao[subject +'+'+ value]
, cv_zhanbi[subject +'+'+ value], p_zhanbi[subject +'+'+ value], plot3_zhanbi[subject +'+'+ value]

def abnormal_output(xubao_data, zhanbi_data):
    prediction_xubao = {}
    prediction_zhanbi = {}
    plot1_xubao = {}
    plot1_zhanbi = {}
    plot2_xubao = {}
    plot2_zhanbi = {}
    params = {}
    data0 = pd.DataFrame()
    for subject, features in xubao_data.items():
        for value in features:
            prediction_xubao[subject +'+'+ value], plot1_xubao[subject +'+'+ value], plot2_xubao[subject +'+'+ value] = build_prophet(xubao_data[subject][value][['ds', 'y']],'2021-05-17', 'logistic', 7, 0.001, 0.01, holidays, subject +'+'+ value,[]) # fit model
            prediction_zhanbi[subject +'+'+ value], plot1_zhanbi[subject +'+'+ value], plot2_zhanbi[subject +'+'+ value] = build_prophet(zhanbi_data[subject][value][['ds', 'y']],'2021-05-17', 'logistic', 7, 0.001, 0.01, holidays, subject +'+'+ value,[]) # fit model
            data1 = abnormal_detection(prediction_xubao[subject +'+'+ value],prediction_zhanbi[subject +'+'+ value], 1, xubao_data[subject][value][['ds', 'y']], zhanbi_data[subject][value][['ds', 'y']], subject, value)
            data0 = pd.concat([data0, data1])
    return data0

data0

a = abnormal_output(xubao_city, zhanbi_city)
b = abnormal_output(xubao_stage, zhanbi_stage)
c = abnormal_output(xubao_agematch, zhanbi_agematch)
d = abnormal_output(xubao_buyway, zhanbi_buyway)
e = abnormal_output(xubao_userfrom, zhanbi_userfrom)
f = abnormal_output(xubao_xiangzhen, zhanbi_xiangzhen)

# 输出最终异常检测
abnormal_output = pd.concat([a, b, c, d, e, f])
abnormal_output.to_excel("异常检测0517.xlsx")  


```
