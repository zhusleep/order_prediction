import xgboost as xgb
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def weekday_workhard():
    w_w = pd.DataFrame([1]*1461, columns=['workhard'], index=pd.date_range('20130101', '20161231') )
    w_w['weekday'] = ''
    i = 0
    temp = []
    for item in w_w.index:
        temp.append(item.weekday())
        #w_w['weekday'][i] = item.weekday()
        #i += 1
    w_w['weekday'] = temp
    # 每隔7天设置一个断点
    start_week_point = [0]
    for j in range(len(w_w)):
        if w_w['weekday'][j] == 0:
            start_week_point.append(j)

    return start_week_point

#面向对象编程
class Route():
    def __init__(self, order, start_loc, end_loc):
        self.order = order
        self.start_loc = start_loc
        self.end_loc = end_loc
        self.single_route = start_loc + '-' + end_loc

def weeksale_static(ts, start_week_point):
    start_week_point = start_week_point
    #############################################
    # 统计周销量及周工作量
    ts['week_sales'] = ''
    p = 0
    real_sales = []
    real_sales_time = []
    temp = []
    for j in range(len(ts)):
        if j < start_week_point[p + 1]:
            temp.append(sum(ts['order_flow'][start_week_point[p]:(start_week_point[p + 1])]))
            #ts['week_job'][j] = sum(ts['workhard'][start_week_point[p]:(start_week_point[p + 1])])
        elif p + 1 != len(start_week_point) - 1:
            p += 1
            temp.append(sum(ts['order_flow'][start_week_point[p]:start_week_point[p + 1]]))
            #ts['week_job'][j] = sum(ts['workhard'][start_week_point[p]:(start_week_point[p + 1])])
            #real_job.append(ts['week_job'][j])
            real_sales.append(sum(ts['order_flow'][start_week_point[p]:start_week_point[p + 1]]))
            real_sales_time.append(ts.index[j])
        else:
            temp.append(sum(ts['order_flow'][start_week_point[p + 1]::]))
            #ts['week_job'][j] = sum(ts['workhard'][start_week_point[p + 1]::])
    ts['week_sales'] = temp
    sales_week = pd.Series(real_sales, index=real_sales_time)
    #jobs_week = pd.Series(real_job, index=real_sales_time)
    return sales_week

def data_clean(path):
    order_flow = pd.read_csv(path, delimiter=',', header=None,encoding='utf-8',
                             names=['city_depart', 'city_desti', 'order_date', 'vehicle_type', 'order_flow'])
    order_flow['route'] = order_flow['city_depart'] + '-' + order_flow['city_desti']
    order_flow.drop(['vehicle_type', 'city_depart', 'city_desti'], axis=1)
    order_flow_new = order_flow.groupby(['route', 'order_date']).sum()  # drop vehicle_type and rearrange the DataFrame
    return order_flow_new.reset_index()

def feature_data(week_sale,year_sale):
    # 对输入数据转化为相应的特征数据
    # 特征集：{前一周销量，前二周销量，前四周销量，当月第几周，上月销量，去年当月销量，过去4周销量方差，过去两个月销量同比，上周末销量}
    # 前一周销量的和
    one_week_before = list(week_sale.shift(1))
    # 前二周销量的和
    two_week_before = list(week_sale.shift(2) + week_sale.shift(1))
    # 前四周销量的和
    temp = []
    for i in range(len(week_sale)):
        try:
            temp.append(sum(week_sale[i-4:i]))
        except:
            temp.append(None)
    four_week_before = temp
    # 当月第几周 待确认数值属性还是类别属性？
    temp = []
    for item_index, item in zip(week_sale.index, week_sale):
        d = item_index.day
        count_week = (d-1)//7+1
        temp.append(count_week)
    current_week = temp
    # 上月销量
    temp = []
    for item_index, item in zip(week_sale.index, week_sale):
        if item_index.month == 1:
            month_date = str(item_index.year - 1) + '-' + '12'
        else:
            month_date = str(item_index.year) + '-' + str(item_index.month-1)
        try:
            temp.append(sum(year_sale[month_date]))
        except:
            temp.append(None)
    one_month_before = temp
    # 去年当月销量
    temp = []
    for item_index, item in zip(week_sale.index, week_sale):
        month_date = str(item_index.year-1) + '-' + str(item_index.month)
        try:
            temp.append(sum(year_sale[month_date]))
        except:
            temp.append(None)
    one_year_before = temp
    # 过去四周销量方差
    temp = []
    for i in range(len(week_sale)):
        try:
            temp.append(np.array(week_sale[i-4:i]).var())
        except:
            temp.append(None)
    var_four_week_before = temp
    # 过去两个月销量同比
    temp = []
    for item_index, item in zip(week_sale.index, week_sale):
        month_date_1 = item_index - relativedelta(months=+2)
        month_date_2 = item_index - relativedelta(months=+1)
        ly_month_date_1 = item_index - relativedelta(months=+2+12)
        ly_month_date_2 = item_index - relativedelta(months=+1+12)

        month_date_1 = str(month_date_1.year) + '-' + str(month_date_1.month)
        month_date_2 = str(month_date_2.year) + '-' + str(month_date_2.month)
        ly_month_date_1 = str(ly_month_date_1.year) + '-' + str(ly_month_date_1.month)
        ly_month_date_2 = str(ly_month_date_2.year) + '-' + str(ly_month_date_2.month)

        try:
            the_sales = sum(year_sale[month_date_1:month_date_2])
            ly_sales = sum(year_sale[ly_month_date_1:ly_month_date_2])
            get_exception = year_sale[ly_month_date_1]
            temp.append((ly_sales-the_sales)/the_sales)
        except:
            temp.append(None)
    increase_last_two_month = temp
    # 上周末销量
    temp = []
    for item_index, item in zip(week_sale.index, week_sale):
        last_sat_date = item_index - datetime.timedelta(days=2)
        last_sun_date = item_index - datetime.timedelta(days=1)
        try:
            temp.append(sum(year_sale[last_sat_date:last_sun_date]))
        except:
            temp.append(None)
    last_weekends_sale = temp
    feature_pd = pd.DataFrame(index=sales_week.index)
    feature_pd['one_week_before'] = one_week_before
    feature_pd['two_week_before'] = two_week_before
    feature_pd['four_week_before'] = four_week_before
    feature_pd['current_week'] = current_week
    feature_pd['one_month_before'] = one_month_before
    feature_pd['one_year_before'] = one_year_before
    feature_pd['var_four_week_before'] = var_four_week_before
    feature_pd['increase_last_two_month'] = increase_last_two_month
    feature_pd['last_weekends_sale'] = last_weekends_sale
    return feature_pd

if __name__ == '__main__':
    path1314 = './data/LSV_order_flow_1314.csv'
    path15 = './data/LSV_order_flow_15.csv'
    path16 = './data/LSV_order_flow_16.csv'

    order_1314 = data_clean(path1314)
    order_flow_new = data_clean(path15)
    order_flow_test_new = data_clean(path16)

    order_flow_1516 = pd.concat([order_1314, order_flow_new, order_flow_test_new])
    order_flow_1516 = order_flow_1516.groupby(['route', 'order_date']).sum()
    order_flow_1516 = order_flow_1516.reset_index()

    #获取路由集合######################################

    Router = set()
    Router_Name = order_flow_1516['route'].unique()
    for single_route in Router_Name:
        #uroute = unicode(single_route, "UTF-8")
        uroute = single_route
        if uroute[:6] == u'上海-上海市':
        #if uroute == u'上海-湖南省':
            order = order_flow_1516[order_flow_1516.route == single_route]
            print(order)
            Router.add(Route(order, uroute[:2], uroute[3:]))
    #填补零订单#########################################

    for item in Router:
        item.order['order_date'] = item.order['order_date'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))
        item.order.set_index(item.order['order_date'], inplace=True)
        item.order = item.order.reindex(index=pd.date_range('20130101', '20161231'), fill_value=0).drop(['order_date'], axis=1)
        fill_route = item.start_loc + '-' + item.end_loc
        item.order.loc[:, ['route']] = fill_route
        print(item.order)
    #pd.concat(objs, axis=0)
    #获取发往全国订单分布###############################
    week_output = []
    start_week_point = weekday_workhard()
    k=0
    for item in Router:

        router_name = item.start_loc + '-' + item.end_loc
        #router_name = encode(router_name,'UTF-8')
        print ('路由', router_name)
        sales_week = weeksale_static(item.order, start_week_point)
        sales_week.name = router_name
        week_output.append(sales_week)
        week_all = pd.concat(week_output,axis=1)
        #print sales_week
    # 提取数据特征
    for r in Router_Name:
        for j in Router:
            if j.single_route == r:
                year_sale = j.order['order_flow']
                feature_pd = feature_data(week_all[r],year_sale)
                print(feature_pd)
    # 设置训练数据和测试数据分割点
    #division_point = '2016-01-01'
    #   for
        #start_date = '2013-03-01'
        #effect_data = week_sale[week_sale.index > start_date]


'''
# load data
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
'''

''''
# read in data
dtrain = xgb.DMatrix('demo/data/agaricus.txt.train')
dtest = xgb.DMatrix('demo/data/agaricus.txt.test')
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
preds = bst.predict(dtest)
print(preds)
'''




