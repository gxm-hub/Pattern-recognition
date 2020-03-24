import cv2
import numpy as np
import matplotlib.pyplot as plt

#训练集{0_9}各100张,测试集各20张
train_number = 100
test_number = 20

#图片尺寸大小
row = 28
col = 28

#训练结果
result = {}



#特征提取
def feature_extration (img_path):
    threhold =127
    img_BGR = cv2.imread(img_path)

    # 转换灰度化单通道
    img_gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    # plt.imshow(img_gray,cmap='gray')
    # plt.show()
    # print(img_gray.shape)  #(28,28)

    # 存储特征值
    feature = np.zeros((7, 7), dtype=np.uint8)
    # feature = []

    # 压缩成7*7
    for i in range(0, 7):
        for j in range(0, 7):
            count = 0   #统计白色像素的个数
            for m in range(0, 4):
                for n in range(0, 4):
                    if int(img_gray[i * 4 + m, j * 4 + n]) > threhold:
                        count += 1
            # 统计白色像素个数
            # print(count)
            # print(f"白色像素所占比例{count/16*100}%")
            file = open('data.txt', 'a')
            file.write(str(count) + ':' + str(count / 16 * 100) + '%')
            file.write('\r\n')
            file.close()
            if count >= 6:
                feature[i, j] = 1
            else:
                feature[i, j] = 0

    #print(feature)
    return(feature)




#图像模板文件化,形成字符串
def moudle_to_ (feature):
        seq=[]
        for i in range(0,7):
            for j in range(0,7):
                seq.append(feature[i,j])
        return (seq)





#相似度比较，欧式距离计算
#将待测试图像的字符串序列，依次与样本字符串序列形成的文件，逐个字符比较，采用欧氏距离，即可得到比较结果，去距离最近的为识别数据。
def forecast(seq):
    forecast = {}
    for label in range(0,10):
        for number in range(0,train_number):
            num = 0
            char_seq = result[str(label)+'_'+str(number)]
            for i in range(0,49):
                num+= np.square(int(char_seq[i])-int(seq[i]))
            forecast[str(label)+'_'+str(number)] = np.sqrt(num)

    forecast_label = 0
    forecast_number = 0
    min_num = 9999

    for label in range(0, 10):
        for number in range(0, train_number):
            if forecast[str(label)+'_'+str(number)]< min_num:
                min_num = forecast[str(label)+'_'+str(number)]
                forecast_label = label
                forecast_number = number

    return forecast_label




#训练特征值存储在result.txt
for label in range(0, 10):
    for number in range(0, train_number):
        img_path = 'C:\\Users\\ocean\\Desktop\\coding_min\\digitRecognition\\image\\train-images\\{0}_{1}.bmp'.format(label, number)
        img_feature = feature_extration(img_path)
        sqe = moudle_to_(img_feature)
        result[str(label)+'_'+str(number)] = sqe

        #模块文件化
        file = open('result.txt', 'a')
        file.write(str(label)+'_'+str(number)+str(result[str(label)+'_'+str(number)]))
        file.write('\r\n')
        file.close()
#print(result)




# 对样本库文件采用同样方式进行模板文件化
label_acc ={}
label_fal = {}
label_rej ={}
for label in range(0,10):
    count = 0
    reject_count = 0
    for number in range(0,test_number):
        test_img_path = 'C:\\Users\\ocean\\Desktop\\coding_min\\digitRecognition\\image\\test-images\\{0}_{1}.bmp'.format(label,number)

        test_feature = feature_extration(test_img_path)
        #plt.imgshow(test_feature,cmap='gray')

        test_sqe = moudle_to_(test_feature)

        #将待测试图像的字符串序列，依次与样本字符串序列形成的文件，逐个字符比较，采用欧氏距离，即可得到比较结果，去距离最近的为识别数据。
        f_num = forecast(test_sqe)
        if f_num == label:
            count += 1
        elif f_num not in range(0,10):
            reject_count +=1
    label_acc[label] = str((count/test_number)*100)+'%'
    label_fal[label] = str(((test_number-count-reject_count)/test_number*100))+'%'
    label_rej[label] = str((reject_count/test_number)*100)+'%'
print("正确识别率:",label_acc)
print("错误识别率:",label_fal)
print("拒绝识别率:",label_rej)




#测试
path = 'C:\\Users\\ocean\\Desktop\\coding_min\\digitRecognition\\image\\test-images\\9_10.bmp'
img_BGR = cv2.imread(path)
# plt.imshow(img_BGR)
# plt.show()

feature = feature_extration(path)
# plt.imshow(feature,cmap='gray')
# plt.show()
sqe = moudle_to_(feature)
print("可能的值为"+str(forecast(sqe)))













