# Chapter 6. 概率论

## 基本定义 (Sec 6.1)

**Sample Space** $\Omega$: 包含随机试验所有可能结果的集合。
例如对于投骰子，$\Omega = \{1, 2, 3, 4, 5, 6\}$。

**Event Space** $\mathcal{A}$：样本空间的所有子集的集合，包括空集和样本空间本身。

**Probability Space** $(\Omega, \mathcal{A}, P)$：包含了样本空间，事件空间和概率的三元组。



**Target Space** $\mathcal{T}$：所有可能的预测结果或标签的集合。它代表了模型输出的所有可能值。元素被称呼为 `state`

$$
P(A)\in [0, 1], P(\Omega) = 1
$$

为避免直接使用概率空间，转而关注感兴趣的量,称为目标空间 $\mathcal{T}$。$\mathcal{T}$ 中的元素被称为 state。

**Random Variable**: $X: \Omega \rightarrow \mathcal{T}$ 是一个函数，将样本空间中的元素映射到目标空间中的元素。

对于一组属于目标空间的情况 $S\subseteq \mathcal{T}$，即其可以定义了 $\{\omega \in \Omega: X(\omega)\in S\}$

如若定义 $X^{-1}: \mathcal{T} \to \Omega$，则有

$$
P_X(S) = P(X \in S) = P(X^{-1}(S)) = P(\{\omega \in \Omega: X(\omega)\in S\})
$$