# FCOS: A Simple and Strong Anchor-free Object Detector

**仅仅记录对19年FCOSv1的补充**

[TOC]



## 1. 介绍

依旧有FCOSv1对于anchor-based模型的劣势分析：超参数的微调；计算量巨大；正负样本差距过大，以及横向比较了别的计算机任务（图像分类，语义分割等）。

> 关于DenseBox的补充：
>
> 现今工作关于anchor-free的探索包括DenseBox：描述了bounding box相对于位置的偏移量。这与对于语义分割的FCN网络相类似，除了每个位置都要回归预测一个4维的vector（左上以及右下坐标）。但是DenseBox为了解决bounding box不同尺寸的问题，它将训练图像resize到几个固定的尺寸，因此DenseBox是创建了一个图像金字塔进行预测，这不符合FCN关于一次性计算所有卷积的特性。

**不同点：**

center-ness的预测是独立于bounding box的回归。



# 2. 我们的方法

**细节与组件**

相对于FCOSv1，正样本的划分方法做出改变：

* FCOSv1：当位置$(x,\,y)$落入由$(\lfloor\frac{s}{2}\rfloor+xs,\, \lfloor\frac{s}{2}\rfloor+ys)$映射回的原图中的ground-truth box中即为正样本。

* FCOSv2：当位置$(x,\,y)$落入由$(\lfloor\frac{s}{2}\rfloor+xs,\, \lfloor\frac{s}{2}\rfloor+ys)$映射回的原图中的ground-truth box**中心区域**中为正样本。

  中心点为$(c_x,\,c_y)$的中心区域的定义为$(c_x-rs,\,c_y-rs,\,c_x+rs,\,c_y+rs)$。其中，$s$为直到当前特征图上的总步长$stride$，$r$则为一个超参数，在COCO数据集中设为$1.5$。

> 一些参数定义的补充说明：
>
> $c^*$：ground-truth box的类标签。背景类别标签为0。
>
> $(l^*,\,r^*,\,r^*,\,b^*)$：一个四维向量vector，代表到当前位置的bounding box的四边距离。是当前位置的回归目标。
>
> 如果位置$(x,\,y)$与bounding box有关联，那么该位置的训练回归目标定义如下：
> $$
> l^*=(x-x^{(i)}_0)/s,\,\,\,\,t^*=(y-y^{(i)}_0)/s\\
> r^*=(x-x^{(i)}_1)/s,\,\,\,\,b^*=(y-y^{(i)}_1)/s\\
> $$
> 其中，除以$s$的意义在于缩小回归目标防止梯度爆炸。



**网络输出**

同FCOSv1.



**损失函数**

在FCOSv1的基础上，**将回归Regression的损失从IoU换成性能更好的GIoU**。



**推理**

同FCOSv1.



**改进：**

详情请见：[FCOS的改进trick](https://zhuanlan.zhihu.com/p/259314634)

1. ![image-20220823143107079](C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220823143107079.png)

2. center-ness作为回归分数的一部分，可以降低远离中心点的bounding box回归权重。

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220823142824379.png" alt="image-20220823142824379" style="zoom: 67%;" />

等等。
