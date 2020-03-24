import cv2
import numpy as np
import matplotlib.pyplot as plt

#训练集{0_9}各100张,测试集各20张
train_number = 100
test_number = 20

#图片尺寸大小
row = 28
col = 28

# pj = np.zeros((10, 49), dtype=float)  # 存储P(x0|wi)~P(X48|Wi)

#特征提取


def feature_extration(img_path):
    threhold = 127
    img_BGR = cv2.imread(img_path)

    # 转换灰度化单通道
    img_gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    # 存储特征值
    feature = np.zeros((7, 7), dtype=np.uint8)
    # 压缩成7*7
    for i in range(0, 7):
        for j in range(0, 7):
            count = 0  # 统计白色像素的个数
            for m in range(0, 4):
                for n in range(0, 4):
                    if int(img_gray[i * 4 + m, j * 4 + n]) > threhold:
                        count += 1

            if count/16 > 0.05:
                feature[i, j] = 1
            else:
                feature[i, j] = 0

    #print(feature)
    return(feature)


#图像模板文件化,形成字符串
def moudle_to_(feature):
    seq = []
    for i in range(0, 7):
        for j in range(0, 7):
            seq.append(feature[i, j])
    return (seq)


#计算先验概率  P(Wi)=Ni/N ,i=0,1,..9
def prior():
    # 存储先验概率
    prior = []
    for label in range(0, 10):
        prior.append(0.1)
    return (prior)


pj = np.zeros((10, 49), dtype=float)  # 存储P(x0|wi)~P(X48|Wi)
x = np.zeros((10, 49), dtype=int)
for label in range(0, 10):
    for number in range(0, train_number):
        img_path = 'C:\\Users\\ocean\\Desktop\\coding_min\\digitRecognition\\image\\train-images\\{0}_{1}.bmp'.format(
            label, number)
        img_feature = feature_extration(img_path)
        sqe = moudle_to_(img_feature)
        for j in range(0, 49):
            if sqe[j] == 1:
                x[label, j] += 1  # 每一类100张图片中每列特征值为1的个数，得到10类49组
for wi in range(0, 10):
    for j in range(0, 49):
        #每一类100张图片中每列特征值为1的概率,即是 Pj(wi),拉普拉斯平滑
        pj[wi, j] = (x[wi, j] + 1) / (train_number + 2)


#计算类条件概率
def conditional(x):  # x 为特征值数组sqe
    p = np.zeros((10, 49), dtype=float)
    px = [1]*10
    for wi in range(0, 10):
        for j in range(0, 49):
            if x[j] == 1:
                p[wi, j] = pj[wi, j]
            elif x[j] == 0:
                p[wi, j] = 1-pj[wi, j]
            px[wi] *= p[wi, j]  # 类条件概率p(x|wi) i=0,1,...9
    #print('类条件概率:',px)
    return(px)


# 后验概率
def get_posterior(p):  # p为类条件概率
    prior = []
    posterior = [1]*10
    # 后验概率 P(Wi|X)= P(Wi)*P(X|Wi)/SUM(P(Wi)*P(X|Wi))

    for wi in range(0, 10):
        prior.append(0.1)
        posterior[wi] = prior[wi]*p[wi]

    for wi in range(0, 10):
        posterior[wi] = posterior[wi]/sum(posterior)
    print('先验概率:', prior)
    print('后验概率：', posterior)


#判断所属的类
def forecast(x):
    houyan = [0]*10
    houyan = conditional(x)
    max_index = houyan.index(max(houyan))
    return max_index


# #训练集
count = np.zeros((10, 49), dtype=np.uint8)
p = [1] * 10
for label in range(0, 10):
    for number in range(0, train_number):
        img_path = 'C:\\Users\\ocean\\Desktop\\coding_min\\digitRecognition\\image\\train-images\\{0}_{1}.bmp'.format(
            label, number)
        img_feature = feature_extration(img_path)
        sqe = moudle_to_(img_feature)
        p = conditional(sqe)


print('条件概率矩阵', pj)
print('类条件概率', p)
get_posterior(p)


# #测试集
label_acc = {}
label_fal = {}
label_rej = {}
houyan = [0]*10
for label in range(0, 10):
    count = 0
    reject_count = 0
    for number in range(0, test_number):
        img_path = 'C:\\Users\\ocean\\Desktop\\coding_min\\digitRecognition\\image\\test-images\\{0}_{1}.bmp'.format(
            label, number)
        img_feature = feature_extration(img_path)
        sqe = moudle_to_(img_feature)
        max_index = forecast(sqe)

        if max_index == label:
            count += 1
        elif max_index not in range(0, 10):
            reject_count += 1
    label_acc[label] = str((count / test_number) * 100) + '%'
    label_fal[label] = str(
        ((test_number - count - reject_count) / test_number * 100)) + '%'
    label_rej[label] = str((reject_count / test_number) * 100) + '%'
print("正确识别率:", label_acc)
print("错误识别率:", label_fal)
print("拒绝识别率:", label_rej)



#测试识别某一张图片
path = 'C:\\Users\\ocean\\Desktop\\coding_min\\digitRecognition\\image\\test-images\\8_3.bmp'
img_BGR = cv2.imread(path)
# plt.imshow(img_BGR)
# plt.show()
houyan = [0] * 10
feature = feature_extration(path)
# plt.imshow(feature,cmap='gray')
# plt.show()
sqe = moudle_to_(feature)
print('判断类为', forecast(sqe))
