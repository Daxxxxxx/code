# 用户画像五边形
## 系统课用户购买力demo  
#### 一、用户范围  
* 基础学科系统课本周在课用户  

#### 二、购买力指标
|  指标   | 库表   | 规则 |
|  :----:  | :----:  | :----: |
| 历史支付金额  |dw_dwd.dwd_eng_order_course_detail_da  | 用户历史所有订单支付金额总和  |
| 历史退款金额  | dw_dwd.dwd_eng_order_course_detail_da | 用户历史所有退款金额总和  |
| 历史支付频次 | dw_dwd.dwd_eng_order_course_detail_da | 用户历史所有订单数量总和 |
| 近30天售卖页浏览次数  | eng.ods_eng_leon_di | /event/ClassIntroWechat/enter  |
| 近30天售卖页停留时长  | eng.ods_eng_leon_di | /time/ClassIntroWechat/duration  |
| 地区房价  | dw_dwd.dwd_conan_house_web_info | 用户所在区的平均房价  |
| 地址数量  | dw_dwd.dwd_conan_house_web_info | 用户地址所在区的数量  |
| 设备数量  | dw_dwd.dwd_eng_user_device_rela_frog_da | 用户历史设备model数量  |
| 设备总价格  | dw_dwd.dwd_eng_user_device_rela_frog_da | 用户历史设备价格总和 |
| 用户类型  | eng.dw_eng_season_user_class_week_info_da | 首购非首购  |
| ~~教育支出~~  | dw_ods.ods_conan_mentor_task_user_label_v2_da | 参课问卷教育支出的答案 |
| 学科规划数量  | dw_ods.ods_conan_mentor_task_user_label_v2_da| 参课问卷用户填写学科规划的科目数  |
#### 三、购买力计算   
* 计算过程：  
    1.根据用户基础特征（无缺失值的特征）计算用户基准购买力  
    2.根据用户额外特征（有缺失值的特征按缺失程度从大到小）依次对基准购买力进行上下浮动  
        - 例：额外特征有地区房价和教育支出，先加入地区房价计算额外购买力排名，对有排名的用户分档加减分；再此基础上再加入教育支出计算额外的教育支出购买力排名，对有排名的用户分档加减分  
        - 有n个额外特征，就需要浮动n次得出最终购买力得分  
        - 加减分数可人工决定
* 指标处理：   
    - 缺失值：不做处理
    - 类别变量：dummy/onehot/labelencoder  
    - 对所有指标进行标准化处理    
    - PCA
* 主成分个数：累计方差贡献率>=85%    
* 指标权重：各主成分的方差贡献率占累计方差贡献率的比值  
* 购买力=wpc1✖️pc1+wpc2✖️pc2+wpc3✖️pc3+... 

#### 四、购买力效果  
1. 系统课应用场景：续报期对用户的差异化沟通（目前是标准化模版）  

    - 购买力值高：金钱成本不是用户主要障碍，策略可以侧重课程价值等其他方面，续报节奏上对营销活动的宣传也可以适当减少

    - 购买力值低：金钱成本是用户的一大障碍，策略需要告诉家长，什么时候买最优惠最合适，刺激用户    
    
2. 购买力如何分层？购买力分层下的续报率是否有显著区分度？

```ipynb
import pandas as pd
import numpy as np
from datetime import datetime

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option("display.max_rows", None)

import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
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
df = spark.sql(f"""

SELECT a.userid, a.subject, a.classid, a.week_dt, a.stage_name
	, a.teacher_ldap, a.stage_weekidx
    , a.is_shougou
    , b.pay_amt_total
	, coalesce(b.refund_amt_total, 0) AS refund_amt_total
    , b.pay_frequency
	, g.is_continued
    , coalesce(c.count, 0) AS page_count
	, coalesce(d.duration, 0) AS page_view_duration
	, e.education_cost
    , coalesce(e.plan_cnt,0) as plan_cnt
	, coalesce(f.address_count, 0) AS address_count
	, f.wealth_potential
    , coalesce(phone_num,0) as phone_num
    , coalesce(device_price,0) as device_price
FROM (
	SELECT userid, subject, classid, week_dt, stage_name
		, orderid, stage_weekidx, teacher_ldap, teacher_leader_ldap, teacher_city
        , if(index > 48, 1, 0) AS is_shougou
	FROM eng.dw_eng_season_user_class_week_info_da
	WHERE dt = date_sub(CURRENT_DATE, 1)
		AND week_dt = '2023-02-06'
		AND subject =1
		AND stage_weekidx BETWEEN 37 AND 48 
    ) a
	LEFT JOIN 
    ( -- 长续
		SELECT *
		FROM eng.dw_eng_long_renew_long_da
		WHERE dt = date_sub(CURRENT_DATE, 1)
			AND season_type != '1_month'
	) g
	ON a.orderid = g.orderid
	LEFT JOIN 
    ( -- 历史订单信息
		SELECT user_id AS userid
        , sum(pay_amt) AS pay_amt_total
        , sum(refund_amt) AS refund_amt_total
		, count(order_id) AS pay_frequency
		FROM dw_dwd.dwd_eng_order_course_detail_da
		WHERE dt = date_sub(CURRENT_DATE, 1)
			AND order_status = '2'
		GROUP BY user_id
	) b
	ON a.userid = b.userid
	LEFT JOIN 
    ( -- 近30天购买页浏览次数
		SELECT userid
			, CASE 
				WHEN other['lessonid'] = '20' THEN 1
				WHEN other['lessonid'] = '46' THEN 2
				WHEN other['lessonid'] = '118' THEN 3
				WHEN other['lessonid'] = '200' THEN 4
				WHEN other['lessonid'] = '282' THEN 7
				WHEN other['lessonid'] = '266' THEN 5
			END AS subject, count(1) AS count
		FROM eng.ods_eng_leon_di
		WHERE dt BETWEEN date_sub(CURRENT_DATE, 29) AND CURRENT_DATE
			AND class_name = 'event'
			AND method_name = 'ClassIntroWechat'
			AND url = '/event/ClassIntroWechat/enter'
			AND other['lessonid'] IN (
				'20', 
				'46', 
				'118', 
				'200', 
				'282', 
				'266'
			)
			AND userid IS NOT NULL
			AND userid != 'null'
			AND userid REGEXP '^[0-9]+$'
			AND other['lessonid'] IS NOT NULL
			AND other['lessonid'] != 'null'
			AND other['lessonid'] REGEXP '^[0-9]+$'
		GROUP BY userid, CASE 
				WHEN other['lessonid'] = '20' THEN 1
				WHEN other['lessonid'] = '46' THEN 2
				WHEN other['lessonid'] = '118' THEN 3
				WHEN other['lessonid'] = '200' THEN 4
				WHEN other['lessonid'] = '282' THEN 7
				WHEN other['lessonid'] = '266' THEN 5
			END
	) c
	ON a.userid = c.userid
		AND a.subject = c.subject
	LEFT JOIN 
    ( -- 购买页浏览时长
		SELECT userid
			, CASE 
				WHEN other['lessonid'] = '20' THEN 1
				WHEN other['lessonid'] = '46' THEN 2
				WHEN other['lessonid'] = '118' THEN 3
				WHEN other['lessonid'] = '200' THEN 4
				WHEN other['lessonid'] = '282' THEN 7
				WHEN other['lessonid'] = '266' THEN 5
			END AS subject, sum(other['duration']) AS duration
		FROM eng.ods_eng_leon_di
		WHERE dt BETWEEN date_sub(CURRENT_DATE, 29) AND CURRENT_DATE
			AND class_name = 'time'
			AND method_name = 'ClassIntroWechat'
			AND url = '/time/ClassIntroWechat/duration'
			AND other['lessonid'] IN (
				'20', 
				'46', 
				'118', 
				'200', 
				'282', 
				'266'
			)
			AND userid IS NOT NULL
			AND userid != 'null'
			AND userid REGEXP '^[0-9]+$'
			AND other['lessonid'] IS NOT NULL
			AND other['lessonid'] != 'null'
			AND other['lessonid'] REGEXP '^[0-9]+$'
		GROUP BY userid, CASE 
				WHEN other['lessonid'] = '20' THEN 1
				WHEN other['lessonid'] = '46' THEN 2
				WHEN other['lessonid'] = '118' THEN 3
				WHEN other['lessonid'] = '200' THEN 4
				WHEN other['lessonid'] = '282' THEN 7
				WHEN other['lessonid'] = '266' THEN 5
			END
	) d
	ON a.userid = d.userid
		AND a.subject = d.subject
	LEFT JOIN 
    (
        select userid 
        , MAX(if(third_label = '教育支出', fourth_label, NULL)) AS education_cost
        , count(DISTINCT if(third_label IN ('英语规划','思维规划', '阅读规划', '美术规划', '写字规划', '音乐规划'), fourth_label, NULL)) AS plan_cnt

        from 
        (
            SELECT b.userid AS userid, b.creator AS creator, b.dbctime AS create_time, to_date(b.dbctime) AS create_dt
                    , b.label_source AS label_source, d.first_show_label AS first_label, d.second_show_label AS second_label, e.showname AS third_label, c.showname AS fourth_label
                    , b.content AS content
                FROM (
                    SELECT string(userid) AS userid, creator, dbctime
                        , to_date(dbctime) AS create_dt, labelid
                        , CASE 
                            WHEN string(userid) = creator THEN '问卷'
                            WHEN source = 10 THEN '问卷'
                            WHEN source = 20 THEN '老师添加'
                            WHEN source = 30 THEN '沟通记录'
                            ELSE '其他'
                        END AS label_source, content, id
                    FROM dw_ods.ods_conan_mentor_task_user_label_v2_da
                    WHERE dt = date_sub(CURRENT_DATE, 1)
                    GROUP BY userid, creator, dbctime, to_date(dbctime), labelid, CASE 
                            WHEN string(userid) = creator THEN '问卷'
                            WHEN source = 10 THEN '问卷'
                            WHEN source = 20 THEN '老师添加'
                            WHEN source = 30 THEN '沟通记录'
                            ELSE '其他'
                        END, content, id
                ) b
                    LEFT JOIN (
                        SELECT id, name, showname, categoryid, type
                            , dbctime
                        FROM dw_ori.ori_conan_mentor_task_label_da
                        WHERE dt = date_sub(CURRENT_DATE, 1)
                    ) c
                    ON b.labelid = c.id
                    LEFT JOIN (
                        SELECT a.showname AS first_show_label, c.showname AS second_show_label, d.labelcategoryid AS third_id
                        FROM (
                            SELECT showname, id
                            FROM dw_ori.ori_conan_mentor_task_label_one_class_da
                            WHERE dt = date_sub(CURRENT_DATE, 1)
                                AND status = '10'
                        ) a
                            LEFT JOIN (
                                SELECT labeloneclassid, labelsecondclassid
                                FROM dw_ori.ori_conan_mentor_task_label_one_label_second_class_da
                                WHERE dt = date_sub(CURRENT_DATE, 1)
                                    AND status = '10'
                            ) b
                            ON a.id = b.labeloneclassid
                            LEFT JOIN (
                                SELECT id, showname, subject, name
                                FROM dw_ori.ori_conan_mentor_task_label_second_class_da
                                WHERE dt = date_sub(CURRENT_DATE, 1)
                            ) c
                            ON b.labelsecondclassid = c.id
                            LEFT JOIN (
                                SELECT labelsecondclassid, labelcategoryid
                                FROM dw_ori.ori_conan_mentor_task_label_second_class_label_category_da
                                WHERE dt = date_sub(CURRENT_DATE, 1)
                                    AND status = '10'
                            ) d
                            ON c.id = d.labelsecondclassid
                    ) d
                    ON c.categoryid = d.third_id
                    JOIN (
                        SELECT id, name, showname
                        FROM dw_ori.ori_conan_mentor_task_label_category_da
                        WHERE dt = date_sub(CURRENT_DATE, 1)
                    ) e
                    ON c.categoryid = e.id
                WHERE e.showname IN ('教育支出','英语规划','思维规划', '阅读规划', '美术规划', '写字规划', '音乐规划')
                    AND d.second_show_label IN ('家庭情况','规划')
            ) as a
            group by userid
	) e
	ON a.userid = e.userid
	LEFT JOIN 
    ( -- 地区房价水平
		SELECT userid
        , count(distinct county) AS address_count
        , sum(price) AS wealth_potential
		FROM 
        (
			SELECT a.userid,a.county, b.price
			FROM 
            (
				SELECT userid,county
				FROM dw_ori.ori_conan_address_delivery_address_da
                group by userid,county
			) a
				LEFT JOIN 
                (
					SELECT city, district, avg(price) AS price
					FROM dw_dwd.dwd_conan_house_web_info
					GROUP BY city, district
				) b
				ON a.county = b.district
		) a
		GROUP BY a.userid
	) f
	ON a.userid = f.userid
    left join 
    ( -- 设备价格/数量
        select user_id as userid
        ,count(distinct model) as phone_num
        ,sum(device_price) as device_price
        from dw_dwd.dwd_eng_user_device_rela_frog_da
        where dt = date_sub(CURRENT_DATE, 1)
        group by user_id
    
    ) as l on a.userid=l.userid


""").toPandas()
df.head(10)
df.info()
df['device_price']=df['device_price'].astype('float64')

x=df[(~df.wealth_potential.isnull())&(~df.device_price.isnull())]['wealth_potential'].values
y=df[(~df.wealth_potential.isnull())&(~df.device_price.isnull())]['device_price'].values

np.corrcoef(x, y)

plt.scatter(x,y, alpha=0.5,c='red')
plt.show()
df.groupby('education_cost').agg({'wealth_potential':'mean'})
df_1=df[(df.education_cost!='未填写')&(df.device_price!='未知')]
df_1['device_price']=df_1['device_price'].astype('float64')
df['wealth_potential'].hist(bins=130,grid=False,alpha=0.5,color='green')
plt.show()
### PCA  计算基准购买力：
df.info()
df.columns
df_base=df[df.columns[~df.isna().any()]]
df_base.info()
len(df_base.columns)
len(df.columns)
df_base.columns
df_base_pca=df_base[['pay_amt_total', 'refund_amt_total',
       'pay_frequency', 'page_count', 'page_view_duration','plan_cnt',
       'address_count', 'phone_num', 'device_price']]
def pcaPlot(df,n):
    x = StandardScaler().fit_transform(df)
    pca = PCA(n_components=n) #选取n个主成分
    pc = pca.fit_transform(x) #对原数据进行pca处理
    print("explained variance ratio: %s" % pca.explained_variance_ratio_) #输出各个主成分所占的比例
    plt.plot(range(1, 9), np.cumsum(pca.explained_variance_ratio_),marker="o",color='red') #绘制主成分累积比例图
    plt.xlim(0, 9)
    plt.ylim(0.1, 1.02)
    plt.show()
pcaPlot(df_base_pca,8)
#观察累计方差解释度，选择7个主成分进行降维
def dimensionReduction(df,df1,n,columns):
    pca = PCA(n_components=n) #选取4个主成分
    x = StandardScaler().fit_transform(df)
    pc = pca.fit_transform(x) 
    pc_df = pd.DataFrame(pc, columns=columns)
    pc_df['userid'] = df1['userid'] 
    explained_variance = pca.explained_variance_ratio_
    variance=pca.explained_variance_
    component = pca.components_

    return pc_df,explained_variance,variance,component
pc_df_base,explained_variance_base,variance_base,component_base = dimensionReduction(df_base_pca,df_base,7,['pc_1', 'pc_2','pc_3','pc_4', 'pc_5','pc_6','pc_7'])
print('输出概览：')
pc_df_base.head(4)
print('主成分向量：\n',component_base)
print('累计方差解释度：\n',sum(explained_variance_base))
特征影响因素：支付金额and设备价格 -> 品牌认知
component_base.T * np.sqrt(variance_base)
* 购买力=wpc1✖️pc1+wpc2✖️pc2+wpc3✖️pc3+...
pc_df_base['购买力基准指数']=pc_df_base.apply(lambda x: (np.dot(explained_variance_base,x[:-1]))/sum(explained_variance_base),axis=1)
pc_df_base.head(4)
def purchasePower(pc_df,df_ori,bins):
    df_ori['购买力基准指数']=pc_df['购买力基准指数']
    df_ori['购买力基准指数_分桶'] = pd.cut(df_ori['购买力基准指数'],bins)
    df_ori['购买力排名'] = df_ori.购买力基准指数.rank(method='first',ascending=False)
    # 购买力标准化

    #df_ori['top']=df_ori.apply(lambda x: 'top' if x['顺序排名']<=rank else '普通',axis=1)
    #df_ori['续报增益_bins'] = pd.cut(df_ori['f.avg_season_rate'],bins)
    #strategy_effect = pd.DataFrame(df_ori.groupby(['续报增益_bins','top'])\
                #.agg({'a.ldap':'count','user_num':'sum','renew_rate':'mean'})).reset_index()
    return df_ori.sort_values(by=['购买力排名'],ascending=True)
df_base_result=purchasePower(pc_df_base,df_base,10)
df_base_result.head(10)
df_base_result.tail(10)
* 购买力基准指数标准化 0～100
def map(data,MIN,MAX):
    d_min = df_base_result['购买力基准指数'].min()   
    d_max = df_base_result['购买力基准指数'].max()
    return (MIN +(MAX-MIN)/(d_max-d_min) * (data - d_min))*10
df_base_result['购买力值']=df_base_result.apply(lambda x: map(x['购买力基准指数'],0,100),axis=1)
df_base_result.head(10)
plt.boxplot(df_base_result['购买力值'][10:].values)
plt.title('Purchase Power Distribution')
plt.show()
plt.hist(df_base_result['购买力值'][10:].values,bins=100,color='red')
plt.title('Purchase ')
plt.show()
举例两个用户：
import random
user_A=random.choice(range(df_base_result.shape[0]))
user_B=random.choice(range(df_base_result.shape[0]))
df_base_result.iloc[user_A,7:17]
df_base_result.iloc[user_B,7:17]
name_list = ['is_shougou','pay_amt_total', 'refund_amt_total',
       'pay_frequency', 'page_count', 'page_view_duration','plan_cnt',
       'address_count', 'phone_num', 'device_price']
user_list = df_base_result.iloc[user_A,7:17].values.tolist()
user_list1 = df_base_result.iloc[user_B,7:17].values.tolist()
print('userA和userB的购买力值为：',df_base_result.iloc[user_A,-1],df_base_result.iloc[user_B,-1])
plt.figure(figsize=(20, 4))
plt.bar(range(len(user_list)), user_list, label='userA',fc = 'grey')
plt.bar(range(len(user_list)), user_list1, bottom=user_list, label='userB',tick_label = name_list,fc = 'black')
plt.legend()
plt.show()
x= df_base_result[['is_shougou','pay_amt_total', 'refund_amt_total',
       'pay_frequency', 'page_count', 'page_view_duration','plan_cnt',
       'address_count', 'phone_num', 'device_price']]
import plotly.express as px
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(x)*10
X_minmax[user_A].tolist()[1:]
X_minmax[user_B].tolist()[1:]
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
categories=['pay_amt_total', 'refund_amt_total',
       'pay_frequency', 'page_count', 'page_view_duration','plan_cnt',
       'address_count', 'phone_num', 'device_price']
fig = make_subplots(rows = 1,cols =2)
fig.add_trace(go.Scatterpolar(
      r=X_minmax[user_A].tolist()[1:],
      theta=categories,
      name='user A'
))
fig.add_trace(go.Scatterpolar(
      r=X_minmax[user_B].tolist()[1:],
      theta=categories,
      name='user B'
))
fig.show()
基准购买力**DONE!!!**
---
Next:根据额外指标浮动
```
