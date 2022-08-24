# 通过ATSS自适应训练样本选择缩小anchor-based和anchor-free差距

[TOC]

## 1. 摘要与介绍

### 1.1. 摘要

anchor-based和anchor-free的主要差别在于**正负样本的划分**。如果anchor-free和anchor-based都使用同一套正负样本划分标准，那么无论是point还是box回归没有什么差距。

于是提出ATSS自适应训练样本选择（Adaptive Training Sample Selection），**根据物体的统计特征自动选择正负样本**。

并在最后讨论了在图像上每个位置**铺设多个锚点来检测物体的必要性**。



### 1.2. 关于anchor-free检测器

anchor-free检测器以两种方式直接找到没有预设anchor box的物体：

* **keypoint-based**：首先定位几个预设或者自我学习到的关键point，然后约束物体的空间范围（CornerNet）。

* **center-based**：利用物体的中心点或区域来定义positive目标，然后预测positive目标到物体边界的四周距离（FCOS）。center-based更类似于anchor-based，它将点作为预设样本而不是anchor box。

anchor-free检测器的好处：能够消除与anchor box有关的超参数，并能够取得与anchor-based检测器相当的性能，使其在泛化能力方面更具潜力。



### 1.3. anchor-based和anchor-free区别

以one-stage anchor-based检测器RetinaNet和center-based anchor-free检测器FCOS为例，他们有**三个区别**：

|                  | RetinaNet（one-stage/anchor-based） | FCOS（center-based/anchor-free）                   |
| ---------------- | ----------------------------------- | -------------------------------------------------- |
| anchor的数量     | 每个位置铺设几个anchor box          | 每个位置铺设一个anchor point                       |
| 对正负样本的划分 | IoU值划分正负样本                   | 利用spatial与scale的限制 <a href="#*">*</a>（FPN） |
| 回归的起始状态   | 从预设的anchor box回归bounding box  | 从anchor point回归bounding box                     |

> **<a name="*">\* </a>关于“利用spatial与scale的限制”**
>
> > **FCOS首先用spatial约束来寻找空间维度的候选正样本，然后使用scale约束来选择尺度维度的最终正样本。**
>
> spatial约束是指从所有特征图point约束到所有落入ground-truth box范围内的anchor point。
>
> scale约束是指FPN生成的每层特征图约束了每层检测物体尺寸的范围。具体来说，对于一层特征图，目标的正样本的尺寸必须满足在该层划定的检测尺寸，除此之外均为负样本。
>
> 详情请见FCOS论文。

经过复现实验证明，如果将RetinaNet和FCOS的正负样本划分控制一致，在性能上并没有多大差距。

由此，本文提出一个新的自适应训练样本选择（ATSS），根据物体的特征自动选择正负样本。



### 1.4. 本工作主要贡献

* 表明anchor-free和anchor-based检测器之间的本质区别实际上是如何定义正负训练样本。

* 提出一种自适应训练样本选择，根据对象的统计特征自动选择正负样本训练。

* 证明在图像上每个位置堆砌多个anchor来检测物体时一种无用的操作。

* 获得MS COCO的SOTA。



## 2. anchor-based和anchor-free检测器的差别分析

采用anchor-based的RetinaNet和anchor-free的FCOS来分析差异。

在第二节中，重点讨论对**正负样本的划分**和**回归的起始状态**。

在第三节中，将讨论**anchor的数量**对于性能的影响。



### 2.1. 实验设置

**数据集**

使用包含80个类别的MS COCO。

**训练细节**

* backbone：使用ImageNet预训练的ResNet50的5层特征金字塔结构。对于RetinaNet，每层特征金字塔都有一个尺寸为$8S$的正方形anchor相关联，$S$为总共的步长stride大小。

* 图像大小：短边为800，长边为1333
* 使用随机梯度下降SGD进行了90k迭代
* 动量：0.9
* 权重衰减：0.0001
* batch size：16

* 学习率：初始为0.01，在60k和80k次迭代时分别衰减0.1


**推理细节**

图像大小与训练一致。score设置为0.05来过滤掉大量背景bounding box。非最大抑制（NMS）设置为0.6。



### 2.2. 消除不一致

**RetinaNet (#A=1)**：只有一个anchor box的anchor-based RetinaNet。基本与FCOS相同，但是FCOS在$AP$性能上大大优于RetinaNet (#A=1)。FCOSv1 : RetinaNet = 37.1% : 32.5%。

**FCOS**：在v2版本上有一些对于v1的优化trick：将centerness移至Regression分支预测；使用GIoU损失函数；通过相应步长stride将Regression目标归一化。FCOSv2达到$AP=37.8\%$。

**分析**

FCOSv1对比RetinaNet来说，有一些通用性的改进trick：

* head添加GroupNormalization (GN)；
* 使用GIoU Regression损失函数；
* 限制ground-truth box的正样本；
* 引入centerness；
* 添加一个可训练的标量。

而这些改进也可以应用于anchor-based的RetinaNet (#A=1)。因此这不是anchor-based和anchor-free的本质区别。

于是将这些FCOSv1的trick应用到ReinaNet中，以排除这些不一致。结果如Table 1所示。

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220824102718975.png" alt="image-20220824102718975" style="zoom:35%;" />

可见，全部应用FCOSv1 trick的RetinaNet (#A=1)依旧有$AP=0.8\%$的差距。由此可以公平地探索他们的本质差异。



### 2.3. 本质差异

排除了通用性trick，现在只剩两个差异：

* 定义正负样本的划分方式；
* 从anchor box还是从anchor point开始Regression。

**定义正负样本的划分方式**

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220824133145607.png" alt="image-20220824133145607" style="zoom:35%;" />

RetinaNet从每层特征金字塔层级选择出anchor box与ground-truth $IoU>{\theta}_p$作为正样本，$IoU<{\theta}_n$作为负样本，然后忽略别的anchor box，对正负样本进行训练。即，RetinaNet利用IoU同时从spatial与scale上直接选择最终阳性。

而FCOS利用spatial与scale的限制 <a href="#*">*</a>来划分不同层的特征金字塔的anchor point。

如果将RetinaNet和FCOS选择的正负样本划定做一次交叉实验，实验结果如上图Table 2。即可发现，定义正负样本的划分方式对提高性能是本质差异之一。

**从anchor box还是从anchor point开始Regression**

依旧是从上图实验结果所示，到底是从anchor box还是anchor point开始Regression是一个无关紧要的点。

**结论**

定义正负样本的划分方式是本质区别。



## 3. 自适应训练样本选择ATSS

### 3.1. 介绍

关于超参数：anchor-based的IoU与之和anchor-free的scale范围是敏感的超参数，不同超参数设置将产生非常不同的结果。

ATSS可以不需要任何超参数就能根据物体的统计特征自动化分出正负样本。

ATSS具体算法流程图如下：

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220824143430259.png" alt="image-20220824143430259" style="zoom: 80%;" />

**解读：**

* **（Algorithm1：Line 3-6）**

  对于图像上的每个ground-truth box $\mathcal{G}$ ，首先找到它在特征金字塔上**所有特征层**的候选positive样本 $\mathcal{C_g}$ ：于第 $\mathcal{i}$ 层特征图上，从此层所有anchor box  $\mathcal{A_i}$ 中选择 $\mathcal{k}$ 个与 $\mathcal{G}$ 最接近的anchor box。 $\mathcal{k}$ 为超参数（几乎不影响结果）。

  假设有 $\mathcal{L}$ 层金字塔特征图，则一个ground-truth box $\mathcal{G}$ 有 $\mathcal{k}\times\mathcal{L}$ 个候选positive样本 $\mathcal{C_g}$ 。

  > Question：
  >
  > L2 distance是什么意思？

* **（Algorithm1：Line 7-10）**

  Line 7：计算这些候选positive样本 $\mathcal{C_g}$ 与ground-truth box $\mathcal{G}$ 之间的IoU值，即 $\mathcal{D_g}$ .

  Line 8-9：计算 $\mathcal{D_g}$ 的平均值 $\mathcal{m_g}$ 和标准差 $\mathcal{v_g}$ .

  Line 10：可以得到 $\mathcal{G}$ 一个IoU阈值 $\mathcal{t_g}=\mathcal{m_g}+\mathcal{v_g}$.

* **（Algorithm1：Line 11-15）**

  选择这些IoU大于等于阈值 $\mathcal{t_g}$ 作为最终的正样本。

  Line 12表明将正样本的选择限制在ground-truth box的中心区域。

  此外，如果一个anchor box分配给多个ground-truth box，那么选择具有最高IoU的那个。

**Motivation：**

* **根据anchor box跟物体间中心距离选择候选anchor box**。距离越近，IoU越大。

* **使用 $\mathcal{m_g}$ 和 $\mathcal{v_g}$ 之和作为IoU的阈值 $\mathcal{t_g}$ **。一个对象的IoU平均 $\mathcal{m_g}$ 是衡量预设anchor box对这个对象的合适程度。一个对象的IoU标准偏差 $\mathcal{v_g}$ 是衡量哪些层适合检测这个物体的标准。使用 $\mathcal{m_g}$ 和 $\mathcal{v_g}$ 之和作为IoU的阈值 $\mathcal{t_g}$ 可以根据对象的统计特征从适当的金字塔层中为每个对象自适应地选择足够的正样本。

  <img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220824172246651.png" alt="image-20220824172246651" style="zoom:35%;" />

  **解释：**

  如果每个level的IoU都高，那么平均 $\mathcal{m_g}$就会高，这时候需要提高IoU阈值 $t_g$ 来适应均值都高的IoU们。描述这种情况的量便是IoU平均 $\mathcal{m_g}$。

  如果其中一个level的IoU比较高，那么说明这个level里面的基本都是质量高的anchor，所以优先选择这个level的anchor。相应的，IoU阈值 $\mathcal{t_g}$ 也需要提高，以此来筛掉其他level的anchor。描述这种情况的量便是IoU标准偏差 $\mathcal{v_g}$。

  > Ref:
  >
  > [1] [ATSS：论文与源码解读](https://zhuanlan.zhihu.com/p/468015663)

* **保持不同对象之间的公平性**。RetinaNet和FCOS的策略往往对较大的物体有更多的正样本，导致不同物体之间的不公平。而ATSS保证了每个对象都有大约 $0.2*\mathcal{kL}$ 的正样本。



### 3.2. 验证

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220824155228017.png" alt="image-20220824155228017" style="zoom: 35%;" />



### 3.3. 分析

**超参数 $\mathcal{k}$**

进行了几个实验来研究超参数 $k$ 的robustness。实验结果如下所示：

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220824155807166.png" alt="image-20220824155807166" style="zoom:35%;" />

可见， $\mathcal{k}$ 在7到17时基本没有变化。而太小的 $k$ 导致采样不完整导致统计不稳定，太大的 $\mathcal{k}$ 导致夹杂较多的低质量anchor box致使性能稍微下降。但是整体的robustness是较好的， $\mathcal{k}$ 作为超参数可以视为没有影响。

**anchor的大小**

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220824160354282.png" alt="image-20220824160354282" style="zoom:35%;" />

可见anchor大小变化具有robustness。



### 3.4. 对比

<img src="C:\Users\34936\AppData\Roaming\Typora\typora-user-images\image-20220824161216785.png" alt="image-20220824161216785" style="zoom:130%;" />



### 3.5. 讨论

还有一个点没有讨论：**每个位置平铺anchor的数量**。

原始的RetinaNet为每个位置平铺了9个anchor（3种比例的长宽比），标记为RetinaNet (#A=9)。在没有使用ATSS的情况下，RetinaNet (#A=9)要比RetinaNet (#A=1)有更好的性能。结果表明，在传统基于IoU的样本选择策略下，每个位置铺设更多的anchor是有效的。

但是使用ATSS后，铺设更多的anchor并没有多大用处，**需要进一步研究以发现期正确的作用**。

