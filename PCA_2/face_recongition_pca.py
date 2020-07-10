import numpy as np
import cv2 as cv
import os
import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk

IMAGE_SIZE = (50, 50)


# 将图片数据集转化为矩阵，其中矩阵从一列代表一张图片
def createDatabase(path):
    # 查看路径下所有文件
    TrainFiles = os.listdir(path)
    # 计算有几个文件（图片命名都是以 序号.jpg方式）减去Thumbs.db
    Train_Number = len(TrainFiles) - 1
    T = []
    # 把所有图片转为1-D并存入T中
    for i in range(1, Train_Number + 1):
        image = cv.imread(path + '/' + str(i) + '.jpg', cv.IMREAD_GRAYSCALE)
        image = cv.resize(image, IMAGE_SIZE)

        image = image.reshape(image.size, 1)
        T.append(image)
    T = np.array(T)
    # 不能直接T.reshape(T.shape[1],T.shape[0]) 这样会打乱顺序，
    print(T.shape)
    T = T.reshape(T.shape[0], T.shape[1])
    return np.mat(T).T

# 获取特征脸
def eigenfaceCore(T):
    # 把均值变为0 axis = 1代表对各行求均值
    m = T.mean(axis=1)
    A = T - m  # 减去均值
    L = (A.T) * (A)
    print('((((',L.shape)
    #     L = np.cov(A,rowvar = 0)
    # 计算AT *A的 特征向量和特征值V是特征值，D是特征向量
    V, D = np.linalg.eig(L)
    # print('特征值：', V)
    print('特征向量：', D.shape)
    print(A.shape)
    L_eig = []
    for i in range(A.shape[1]):
        #         if V[i] >1:
        L_eig.append(D[:, i])
    L_eig = np.mat(np.reshape(np.array(L_eig), (-1, len(L_eig))))
    # 计算 A *AT的特征向量
    eigenface = A * L_eig
    return eigenface, m, A

# 图像识别
def recognize(testImage, eigenface, m, A):
    _, trainNumber = np.shape(eigenface)
    # 投影到特征脸所表示的空间中
    projectedImage = eigenface.T * (A)
    # 可解决中文路径不能打开问题
    testImageArray = cv.imdecode(np.fromfile(testImage, dtype=np.uint8), cv.IMREAD_GRAYSCALE)
    # 转为1-D
    testImageArray = cv.resize(testImageArray, IMAGE_SIZE)
    testImageArray = testImageArray.reshape(testImageArray.size, 1)
    testImageArray = np.mat(np.array(testImageArray))
    differenceTestImage = testImageArray - m
    projectedTestImage = eigenface.T * (differenceTestImage) # 将用于测试的矩阵投影到新的空间中
    distance = [] # 记录测试图片与用于训练的图片之间的距离
    for i in range(0, trainNumber):
        q = projectedImage[:, i]
        temp = np.linalg.norm(projectedTestImage - q) # 范数
        distance.append(temp)

    minDistance = min(distance) # 找到差距最小的
    index = distance.index(minDistance)
    cv.imshow("recognize result", cv.imread('./TrainDatabase' + '/' + str(index + 1) + '.jpg', cv.IMREAD_GRAYSCALE))
    cv.waitKey()
    return index + 1


# 进行人脸识别主程序
def example(filename):
    T = createDatabase('./TrainDatabase')
    eigenface, m, A = eigenfaceCore(T)
    testimage = filename
    print(testimage)
    print(recognize(testimage, eigenface, m, A))


# 构建可视化界面
def gui():
    root = tk.Tk()
    root.title("pca face")

    # 点击选择图片时调用
    def select():
        filename = tkinter.filedialog.askopenfilename()
        if filename != '':
            s = filename  # jpg图片文件名 和 路径。
            im = Image.open(s)
            tkimg = ImageTk.PhotoImage(im)  # 执行此函数之前， Tk() 必须已经实例化。
            l.config(image=tkimg)
            btn1.config(command=lambda: example(filename))
            btn1.config(text="开始识别")
            btn1.pack()
            # 重新绘制
            root.mainloop()

    # 显示图片的位置
    l = tk.Label(root)
    l.pack()

    btn = tk.Button(root, text="选择识别的图片", command=select)
    btn.pack()

    btn1 = tk.Button(root)  # 开始识别按钮，刚开始不显示
    root.mainloop()


if __name__ == "__main__":
    gui()
