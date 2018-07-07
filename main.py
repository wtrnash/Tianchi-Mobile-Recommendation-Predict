import pandas as pd
import numpy as np
import tensorflow as tf


# 数据预处理
item_data = pd.read_csv("tianchi_fresh_comp_train_item.csv")
user_data = pd.read_csv("tianchi_fresh_comp_train_user.csv")

user_data = user_data.drop_duplicates()  # 去重
# 取交集
user_data = pd.merge(user_data, item_data, on='item_id', how='inner')

# 处理时间
user_data_time = user_data['time'].str.replace('-', '')  # 去除-
user_data_time = user_data_time.str.replace('2014', '')  # 去除2014
user_data['time'] = user_data_time.str.replace(' ', '').astype(np.int64)  # 去除空格
# 取12月17日的数据来预测12月18日数据
user_data_12_17 = user_data[(user_data['time'] >= 121700) & (user_data['time'] <= 121723)]

# 12月18日的数据
user_data_12_18 = user_data[(user_data['time'] >= 121800) & (user_data['time'] <= 121823)]
# 去除18日中不在17日的用户
user_data_12_18 = user_data_12_18[user_data_12_18['user_id'].isin(user_data_12_17['user_id'])]

# 去除18日中不在17日的商品
user_data_12_18 = user_data_12_18[user_data_12_18['item_id'].isin(user_data_12_17['item_id'])]

# 去除17日中不在18日的用户
user_data_12_17 = \
    user_data_12_17[user_data_12_17['user_id'].isin(user_data_12_18['user_id'])]

# 去除17日中不在18日的商品
user_data_12_17 = \
    user_data_12_17[user_data_12_17['item_id'].isin(user_data_12_18['item_id'])]

user_data_12_18 = user_data_12_18[['user_id', 'item_id', 'behavior_type']]
user_data_12_18 = user_data_12_18.drop_duplicates()  # 去重

# 将18日买的就设为1,不买的设为0
user_data_12_18['is_buy'] = 0
user_data_12_18['is_buy'].loc[user_data_12_18['behavior_type'] == 4] = 1
# 将一天内如果有多条记录所以聚合在一起
user_data_12_18 = user_data_12_18.groupby(['user_id', 'item_id']).agg({'is_buy': np.sum})
user_data_12_18['is_buy'].loc[user_data_12_18['is_buy'] > 1] = 1
user_data_12_18 = user_data_12_18.reset_index()
user_data_12_18 = user_data_12_18[['user_id', 'item_id', 'is_buy']]
# 浏览数、收藏数、购买数、加入购物车数
user_data_12_17['num_of_browse'] = 0
user_data_12_17['num_of_browse'].loc[user_data_12_17['behavior_type'] == 1] = 1
user_data_12_17['num_of_collection'] = 0
user_data_12_17['num_of_collection'].loc[user_data_12_17['behavior_type'] == 2] = 1
user_data_12_17['num_of_add_to_cart'] = 0
user_data_12_17['num_of_add_to_cart'].loc[user_data_12_17['behavior_type'] == 3] = 1
user_data_12_17['num_of_buy'] = 0
user_data_12_17['num_of_buy'].loc[user_data_12_17['behavior_type'] == 4] = 1
user_data_12_17 = user_data_12_17.groupby(['user_id', 'item_id']).agg({
    'num_of_browse': np.sum,
    'num_of_collection': np.sum,
    'num_of_add_to_cart': np.sum,
    'num_of_buy': np.sum
})
user_data_12_17 = user_data_12_17.reset_index()

user_data_12_18 = pd.merge(user_data_12_18, user_data_12_17, on=['user_id', 'item_id'], how='inner')

x_data_set = user_data_12_18[['num_of_browse', 'num_of_collection', 'num_of_add_to_cart', 'num_of_buy']]
y_data_set = user_data_12_18[['is_buy']]
x_data_set = pd.DataFrame(x_data_set, dtype=float)

# 模型建立
x = tf.placeholder(tf.float32, [None, 4])
w1 = tf.Variable(tf.random_normal([4, 100], stddev=0.01), name='w1')
b1 = tf.Variable(tf.zeros([100]), name='b1')
hidden1 = tf.sigmoid(tf.matmul(x, w1) + b1, name="sigmoid")
w2 = tf.Variable(tf.random_normal([100, 1], stddev=0.01), name='w2')
b2 = tf.Variable(tf.zeros([1]), name='b2')
y = tf.matmul(hidden1, w2) + b2

y_ = tf.placeholder(tf.float32, [None, 1])
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(5).minimize(cost)
saver = tf.train.Saver()
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# 训练
for i in range(100000):
    sess.run(train_step, feed_dict={x: x_data_set, y_: y_data_set})
    if i % 10 == 0:
        loss = sess.run(cost, feed_dict={x: x_data_set, y_: y_data_set})
        print("After %d training step(s), loss is %g" % (i, loss))

saver.save(sess, './model.ckpt')


# 取把商品加入购物车的记录
# user_data = user_data[user_data.behavior_type.isin([3])]

# 创建12月18日把商品加入购物车的list
# user_item_list = []
# for index, row in user_data.iterrows():
#     if row['time'][0:10] == "2014-12-18":
#         user_item_list.append([row['user_id'], row['item_id']])

# print(user_item_list)
# final_list = []
# 转为str
# for value in user_item_list:
#     final_list.append([str(value[0]), str(value[1])])

# result_df = pd.DataFrame(final_list)  # 转为DataFrame
# result_df = result_df.drop_duplicates() # 去重
# result_df.rename(columns={0: 'user_id', 1: 'item_id'}, inplace=True)
# result_df.to_csv("tianchi_mobile_recommendation_predict.csv", encoding='utf-8', index=None)
