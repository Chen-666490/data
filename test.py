from PIL import Image

# 打开图像
img = Image.open('D:/data_all/data/rice_sick/Bacterialblight/BACTERIALBLIGHT1_176.jpg')

# 打印图像格式
print(img.format)

# 打印图像模式
print(img.mode)