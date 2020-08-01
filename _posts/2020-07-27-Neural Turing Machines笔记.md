---
layout:     post
title:      Neural Turing Machines
subtitle:   学习笔记
date:       2020-07-27
author:     xlc
header-img: img/post-bg-NLP.jpg
catalog: true
tags:
    - NTM
    - NLP
---

# Neural Turing Machines

## 一、参考资料

[记忆网络之Neural Turing Machines](https://zhuanlan.zhihu.com/p/30383994)

[Neural Turing Machines-NTM系列（一）简述](https://blog.csdn.net/rtygbwwwerr/article/details/50548311)

[神经图灵机 Neural Turing Machine-DeepMind](https://www.jianshu.com/p/94dabe29a43b)

[论文(打开花的时间可能比较长)](http://arxiv.org/abs/1410.5401)

## 二、图灵机

来自[wiki百科](https://zh.wikipedia.org/wiki/%E5%9B%BE%E7%81%B5%E6%9C%BA)

用机器来模拟人们用纸笔进行数学运算的过程，他把这样的过程看作下列两种简单的动作：

* 在纸上写上或擦除某个符号；
* 把注意力从纸的一个位置移动到另一个位置；

而在每个阶段，人要决定下一步的动作，依赖于（a）此人当前所关注的纸上某个位置的符号和（b）此人当前思维的状态。

## 三、NTM 神经网络图灵机

就是使用神经网络来实现图灵机之中的读和写操作。

### 3.1 构成

`Controller` 相当于电脑的处理器

`Memory` 相当于电脑的内存

`Read Heads` 对于内存的读磁头，相当于寄存器

`Write Heads` 对于内存的写磁头，相当于寄存器

![a3gaCD.png](https://s1.ax1x.com/2020/08/01/a3gaCD.png)

![a3gRPS.jpg](https://s1.ax1x.com/2020/08/01/a3gRPS.jpg)

### 3.1 读

$M_{t}$是一个矩阵，其中R行表示的是一共有R个内存地址的数目，而C列就表示每一个内存地址的存储的记忆内容大小。

`Controller`会给出一个`attention`的向量，这个向量的长度为R，一般可以写作：$w_{t}$，而每个元素$w_{t}(i)$是$M_{t}$的第i行$M_{t}(i)$的权重，一般会被归一化，因此可以得到：

$$\sum_{i=1}^{R}w_{t}(i) = 1$$

然后我们需要的读磁头`Read Head`就是权重和矩阵的线性组合：

$$ r_{t} := \sum_{i=1}^{R}w_{t}(i)M_{t}(i)$$

### 3.2 写

第一步：擦除(erasing)

引入一个擦除向量$e_{t}$，维度和$w_{t}$相同，都是R。所有的值都在[0,1]区间内，得到一个被擦除后的量的矩阵如下：

$$ M_{t}^{erased}(i):= M_{t-1}(i)[1-w_{t}(i)e_{t}]$$


第二步：添加(add)

记忆矩阵会被一个长度为C的向量$a_{t}$更新：

$$ M_{t}(i):= M_{t}^{erased}(i) + w_{t}(i)a_{t}$$


只有当$w_{t}$和$e_{t}$全部是1的时候，整个`memory`就会被清空。

### 3.3 寻址

读和写的操作关键就是要控制器产生权重矩阵，这个矩阵的产生过程分为以下四步：

#### 3.3.1 content-based addressing

`head`会产生一个长度为C的`key vector` $k_{t}$， 然后用我们熟悉的余弦相似度去计算其和记忆矩阵$M_{t}$的相似性：

$K(u,v) := \frac{u · v}{||u|| · ||v||}$

对$M_{t}$的每一行都进行寻址之后，做一次归一化。

$$w_{t}^{c}(i) := \frac{e^{\beta_{t}K(k_{t},M_{t}(i))}}{\sum_{j}e^{\beta_{t}K(k_{t},M_{t}(j))}}$$

$\beta_{t}$用户控制聚焦的精度，如果它越小，聚焦的范围就越小，这次寻址的记忆范围就越大。

#### 3.3.2 location-based addressing

从特定的内存地址中读写，一般会用一个`interpolation gate` $g_{t} \in (0,1)$ 去把`content weight vector` $w_{t}^{c}(i)$ 和 上一个`head` 产生的 `weight vector` $w_{t}$ 结合起来，利用`interpolation gate` 确定`content-based addressing`的使用频率。

$$w_{t}^{g} := g_{t}w_{t}^{c} + (1 - g_{t})w_{t-1}$$

#### 3.3.3 循环卷积

`interpolation`之后，`head`产生了一个`normalized shift weighting` $s_{t}$，然后我们就要对权重进行旋转位移，要让权重注意从一个位置扩展到周围的位置，一般我们称呼为循环卷积。

$$ \widetilde w_{t}(i) := \sum_{j=0}^{R-1}w_{t}^{g}(j)s_{t}(i-j)$$

具体的例子可以参考这个：

假设移位的范围在-1到1之间（即最多可以前后移动一个位置），则移位值将有3种可能：-1,0,1，对应这3个值，st也将有3个权值。那该怎么求出这些权值呢？比较常见的做法是，把这个问题看做一个多分类问题，在`Controller`中使用一个softmax层来产生对应位移的权重值。在论文中还实验了一种方法：在`Controller`中产生一个缩放因子，该因子为移位位置上均匀分布的下界。

比如，如果该缩放因子值为6.7，那么$s_{t}(6)=0.3$,$s_{t}(7)=0.7$，$s_{t}$的其余分量为0（只取整数索引）。

为了方便卷积，我们将卷积的过程写成了一个矩阵的形式：

![a3ysbV.png](https://s1.ax1x.com/2020/08/01/a3ysbV.png)

#### 3.3.4 锐化

在卷积之后，可能会让权重趋近于均匀化，可能会让本来集中于某个位置的焦点发散，于是我们要进行一次锐化，让注意力能够继续集中于某些位置。

`head`此时要产生一个标量$\gamma \ge 1$:

$$ w_{t}(i) := \frac{\widetilde w_{t}(i)^{\gamma_{t}}}{\sum_{j}\widetilde w_{t}(j)^{\gamma_{t}}}$$

对最后两步，有一个例子可以参考：

![a3gVH0.png](https://s1.ax1x.com/2020/08/01/a3gVH0.png)

因此最后整个流程整理下来，就可以用这张图来表示。可以说非常清晰了。

![https://img-blog.csdn.net/20160228195112020](https://img-blog.csdn.net/20160228195112020)

## 四、优点：有利于泛化到长序列的训练

在传统的神经网络中，一般都是要求在较短的序列长度进行训练会得到比较好的表现，因为传递的深度增加，其在每一步更新过程中所有的隐层向量都会被更新，h的频繁更新，导致其无法记录更长久的信息。而且hidden_size也限制了模型记忆信息的容量。

图一是LSTM泛化到长序列的训练结果：

![a3gIrn.png](https://s1.ax1x.com/2020/08/01/a3gIrn.png)

图二是NTM泛化到长序列的训练结果：

![a3gobq.png](https://s1.ax1x.com/2020/08/01/a3gobq.png)
