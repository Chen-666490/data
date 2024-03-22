import pandas as pd
import shutil
import os

# 读取CSV文件
df = pd.read_csv('D:\\data_all\\data\\strawberry\\nongzuowu\\train.csv')

# 遍历每一行
for index, row in df.iterrows():
    # 获取图片名和类别
    image_name = row['image']
    category = row['label']

    # 构建源文件和目标文件的路径
    src = f'D:\\data_all\\data\\strawberry\\nongzuowu\\train\\{image_name}'
    dst = f'D:\\data_all\\data\\strawberry\\strawberry_train\\{category}\\{image_name}'

    # 创建目标文件夹
    if not os.path.exists(f'D:\\data_all\\data\\strawberry\\strawberry_train\\{category}'):
        os.makedirs(f'D:\\data_all\\data\\strawberry\\strawberry_train\\{category}')

    # 移动文件
    shutil.move(src, dst)
