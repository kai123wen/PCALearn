from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# convert 函数使用：https://blog.csdn.net/icamera0/article/details/50843172
im = np.array(Image.open('testImage.jpg').convert('L'))
height, width = im.shape[0:2]  # height：图片的高；width：图像的宽
points = []
for i in range(height):
    for j in range(width):
        if im[i, j] < 128.0:  # 小于128的灰度值当作黑点取出来
            points.append([float(j), float(height) - float(i)])  # 以图片的左下角为坐标系的原点
print(points)
im_X = np.mat(points).T  # 转置之后，行表示维度（x和y），每列表示一个点（样本）
print('im_X=', im_X, 'shape=', im_X.shape)

"""
# 使用PCA的步骤：
1）将原始数据按列组成 d 行 n 列矩阵 X
重要说明：d对应的就是数据的字段（或叫特征、维），而n表示n条记录（或叫样本），即每1列对应1个样本，之所以行和列这样安排，是为了与数学公式保持一致，很多文章对这一点都没有明确的说明，导致计算的方式各有不同，让人产生不必要的困惑
2）将 X 的每个维（行）进行零均值化，即将行的每个元素减去这一行的均值
3）求出 X 的协方差矩阵 C，即 X 乘 X 的转置
4）求出 C 所有的特征值及对应的特征向量
5）将特征向量按对应特征值大小从上到下按行排列成矩阵，取前 k 行组成矩阵 E
6）Y=EX 即为降维到 k 维后的数据
"""


def pca(X, k=1):  # 降为k维
    d, n = X.shape
    mean_X = np.mean(X, axis=1)  # axis为0表示计算每列的均值，为1表示计算每行均值
    print('mean_X=', mean_X)
    X = X - mean_X
    # 计算不同维度间的协方差，而不是样本间的协方差，方法1：
    C = np.dot(X, X.T)
    e, EV = np.linalg.eig(np.mat(C))  # 求协方差的特征值和特征向量，e为特征值，EV为特征向量
    print('C=', C)
    print('e=', e)
    print('EV=', EV)
    e_idx = np.argsort(-e)[:k]  # 获取前k个最大的特征值对应的下标（注：这里使用对负e排序的技巧，反而让原本最大的排在前面）
    EV_main = EV[:, e_idx]  # 获取特征值（下标）对应的特征向量，作为主成分
    print('e_idx=', e_idx, 'EV_main=', EV_main)
    low_X = np.dot(EV_main.T, X)  # 这就是我们要的原始数据集在主成分上的投影结果
    return low_X, EV_main, mean_X


low_X, EV_main, mean_X = pca(im_X)
print("low_X=", low_X)
print("EV_main=", EV_main)
recon_X = np.dot(EV_main, low_X) + mean_X  # 把投影结果重构为二维表示，以便可以画出来直观的看到
print("recon_X.shape=", recon_X.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(im_X[0].A[0], im_X[1].A[0], s=1, alpha=0.5)
ax.scatter(recon_X[0].A[0], recon_X[1].A[0], marker='*', s=100, c='blue', edgecolors='white')
plt.show()
