## 1.开发环境说明
 - 基于python3.和opencv3.开发，需要安装python的一些库，numpy,matplotlib//或者Anaconda
- 基于模板匹配的手写数字识别，采用mnist部分数据集
-----
## 2.[原理](https://github.com/gxm-hub/Pattern-recognition/blob/master/Min_error_Bayes/%E5%AE%9E%E9%AA%8C%E5%8E%9F%E7%90%86.JPG)
>2.1 基于模板匹配的数字识别，将标准的28*28像素的数字0~9读取，二值化，对每个数字进行等分区域分割压缩7**7，统计每个区域内的白色像素点的个数，即为特征初值.将得到的特征值字符串化0100

>2.2 计算先验概率
     p(wi)= Ni/N  
 每类样本数除以每类样本总数

>2.3 计算条件概率pj(wi)，Ni张样本每列特征值中为1的概率

>2.4 计算类条件概率 

        P（X|Wi）= P(X=a|Wi) a=0,1

> 2.5 贝叶斯公式
------
## 3. 运行
> # git clone


## -- images  
##  --Min_error_Bayes
-----
## 4. [运行结果](https://github.com/gxm-hub/Pattern-recognition/blob/master/Min_error_Bayes/%E5%AE%9E%E9%AA%8C2.png)


