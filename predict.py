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

# 12月18日的数据
user_data_12_18 = user_data[(user_data['time'] >= 121800) & (user_data['time'] <= 121823)]

user_data_12_18['num_of_browse'] = 0
user_data_12_18['num_of_browse'].loc[user_data_12_18['behavior_type'] == 1] = 1
user_data_12_18['num_of_collection'] = 0
user_data_12_18['num_of_collection'].loc[user_data_12_18['behavior_type'] == 2] = 1
user_data_12_18['num_of_add_to_cart'] = 0
user_data_12_18['num_of_add_to_cart'].loc[user_data_12_18['behavior_type'] == 3] = 1
user_data_12_18['num_of_buy'] = 0
user_data_12_18['num_of_buy'].loc[user_data_12_18['behavior_type'] == 4] = 1
user_data_12_18 = user_data_12_18.groupby(['user_id', 'item_id']).agg({
    'num_of_browse': np.sum,
    'num_of_collection': np.sum,
    'num_of_add_to_cart': np.sum,
    'num_of_buy': np.sum
})
user_data_12_18 = user_data_12_18.reset_index()

# 模型建立
x = tf.placeholder(tf.float32, [1, 4])
w1 = tf.Variable(tf.random_normal([4, 100], stddev=0.01), name='w1')
b1 = tf.Variable(tf.zeros([100]), name='b1')
hidden1 = tf.sigmoid(tf.matmul(x, w1) + b1, name="sigmoid")
w2 = tf.Variable(tf.random_normal([100, 1], stddev=0.01), name='w2')
b2 = tf.Variable(tf.zeros([1]), name='b2')
y = tf.sigmoid(tf.matmul(hidden1, w2) + b2)

saver = tf.train.Saver()
sess = tf.InteractiveSession()
saver.restore(sess, './model.ckpt')
answer_list = []
for index, row in user_data_12_18.iterrows():
    result = sess.run(y,  feed_dict={x: [row[
        ['num_of_browse', 'num_of_collection', 'num_of_add_to_cart', 'num_of_buy']]]})
    if result[0] >= 0.5:
        answer_list.append([row['user_id'], row['item_id']])

print(answer_list)
final_list = []
# 转为str
for value in answer_list:
    final_list.append([str(value[0]), str(value[1])])

result_df = pd.DataFrame(final_list)  # 转为DataFrame
result_df = result_df.drop_duplicates() # 去重
result_df.rename(columns={0: 'user_id', 1: 'item_id'}, inplace=True)
result_df.to_csv("tianchi_mobile_recommendation_predict.csv", encoding='utf-8', index=None)
