import pandas as pd

item_data = pd.read_csv("tianchi_fresh_comp_train_item.csv")
user_data = pd.read_csv("tianchi_fresh_comp_train_user.csv")
print(user_data.info())
user_data = user_data.drop_duplicates() # 去重
print(user_data.info())
# 取交集
user_data = pd.merge(user_data, item_data, on='item_id', how='inner')
print(user_data.info())

# 取把商品加入购物车的记录
user_data = user_data[user_data.behavior_type.isin([3])]

# 创建12月18日把商品加入购物车的list
user_item_list = []
for index, row in user_data.iterrows():
    if row['time'][0:10] == "2014-12-18":
        user_item_list.append([row['user_id'], row['item_id']])

print(user_item_list)
final_list = []
# 转为str
for value in user_item_list:
    final_list.append([str(value[0]), str(value[1])])

result_df = pd.DataFrame(final_list)  # 转为DataFrame
result_df = result_df.drop_duplicates() # 去重
result_df.rename(columns={0: 'user_id', 1: 'item_id'}, inplace=True)
result_df.to_csv("tianchi_mobile_recommendation_predict.csv", encoding='utf-8', index=None)
