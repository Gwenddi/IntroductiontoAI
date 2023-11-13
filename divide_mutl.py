import csv
import numpy as np
from matplotlib import pyplot as plt

with open('iris.data') as csv_file:
    data = list(csv.reader(csv_file, delimiter=','))

# 将鸢尾花的三个品种映射到数字0/1/2    
label_map = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica':2
} 

# 抽取样本
X = np.array([[float(x) for x in s[:-1]] for s in data[:150]], np.float32) # X是一个四维数据
Y = np.array([[label_map[s[-1]]] for s in data[:150]], np.float32) 

# 将 Y 转换为可以用0、1表示的向量
tmp = np.zeros((Y.shape[0],3)) # 寄存器
for i in range(Y.shape[0]):
    if Y[i] == 0:
        tmp[i] = [1,0,0]
    if Y[i] == 1:
        tmp[i] = [0,1,0]
    if Y[i] == 2:
        tmp[i] = [0,0,1]
Y = tmp

# 分割数据集

# 将数据集按照8：2划分为训练集和测试集
train_idx = np.random.choice(150, 120, replace=False)

test_idx = np.array(list(set(range(150)) - set(train_idx)))

X_train, Y_train = X[train_idx], Y[train_idx]
b = np.ones((X_train.shape[0],1))
X_train = np.hstack((X_train, b))
X_test, Y_test = X[test_idx], Y[test_idx]
b = np.ones((X_test.shape[0],1)) # 添加常数项
X_test = np.hstack((X_test, b))

# 决策函数 
def softmax(x,w):
    x = x.reshape(x.shape[0],1) # 转化为列向量
    a = (w.T)@x
    a = a - max(a) # 防止数据上溢出
    return np.exp(a)/(np.sum(np.exp(a)))


# 交叉熵损失函数 
def loss(x,y,w):
    sum = 0; # 初始化

    for i in range(x.shape[0]):
        y_hat = softmax(x[i],w) # 预测值
        sum += np.dot(y[i],np.log(y_hat))

    return -sum/x.shape[0] # 求均值

# 梯度函数 
def gradient(x,y,w):
    # 这里注意，一维数组无法进行转置，只能先变成二维数组
    y_hat = softmax(x,w) # 预测值
    y = y.reshape(y.shape[0],1) # 变为二维矩阵
    error = (y-y_hat)
    x = x.reshape(x.shape[0],1)
    return -x @ error.T # 返回该样本点所在的梯度值
    
# 训练函数 
def train(x,y,w,lr=0.05,epoch=300): # 学习率是0.05,最大的迭代次数是epoch=300
    train_err = []
    test_err = []
    for i in range(epoch):
        reg = np.zeros((w.shape[0],w.shape[1])); # 存储梯度值的寄存器初始化
        if loss(x,y,w) > 0:
            for j in range(x.shape[0]):
                reg += gradient(x[j],y[j],w) # 获得所有样本梯度的累加
            reg = reg/x.shape[0] # 获得梯度均值
            w = w - lr*reg # 损失值大于0，计算梯度，更新权值
        test_err.append(test(X_test, Y_test,w))
        train_err.append(test(X_train,Y_train,w))
        # print('epoch:',i,'train error:',train_err[-1],'test error:',test_err[-1])
    return w,train_err,test_err

# 定义测试函数 
def test(x,y,w):
    right = 0
    for i in range(x.shape[0]):
        max = np.argmax(softmax(x[i],w)) # 最大值所在位置
        max_y = np.argmax(y[i]) # 找到y中1的位置，就是所属的分类类别
        if max == max_y:
            right += 1
    return 1- right/x.shape[0]


w = np.ones((X_train.shape[1],Y_train.shape[1]))
w,train_err,test_err = train(X_train,Y_train,w) 
print(w)
print('最终训练误差和测试误差','train error:',train_err[-1],'test error:',test_err[-1])

# 绘制训练误差 train error
plt.plot(train_err)
plt.title('Softmax')
plt.xlabel('epoch')
plt.ylabel('train error')
plt.ylim((-0.3, 1))
plt.grid()
plt.show()

# 绘制测试误差test error
plt.plot(test_err)
plt.title('Softmax')
plt.xlabel('epoch')
plt.ylabel('test error')
plt.ylim((-0.3, 1))
plt.grid()
plt.show()