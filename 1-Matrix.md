# Lecture 1：矩阵

## 相容（consistent）与不相容（inconsistent）

方程组就是无解被称为不相容（inconsistent system）  
有解被称为相容（consistent system）

## 线性相关性（Linear dependence）

对于向量集合 $A = \left\{v_1, v_2, ..., v_p \right\}, v \in \mathbb{R}^n$，如果其中一个向量可以表示为其他向量的线性组合，那么这个向量就是线性相关的。即：

$$
x_1 v_1 + x_2 v_2 + \cdots + x_p v_p = 0
$$

iff 其有平凡解（即 $x_1 = x_2 = \cdots = x_p = 0$），$A$ 是线性无关。

否则线性相关。

如有非平凡解，即表示一个向量可以被其他向量线性组合表示，那么这个向量就是线性相关的。

> $A = \left\{v_1, v_2, ..., v_p \right\}$，$Ax=0$ 只有零解 $\Longleftrightarrow$ $A$ 是线性无关
> 
> $A= \left\{v_1\right\}$，$A$ 是线性无关的 $\Longleftrightarrow$ $v_1 \neq 0$
> 
> $A= \left\{v_1, v_2\right\}$，$A$ 是线性相关的 $\Longleftrightarrow$ $v_1 = kv_2$
> 
> $|A| = p, v\in \mathbb{R}^d$，如果 $p > d$，那么 $A$ 是线性相关的。 *（抽屉原理）*
>
> $|A| = 0$ $\Longleftrightarrow$ 线性相关




## 系数矩阵（Coefficient Matrix）与增广矩阵（Augmented Matrix）

如果考虑线性方程 $2x + 3y = 5$ 和 $3x - 2y = 7$，可以将其写成矩阵形式：

$$
\begin{bmatrix}
2 & 3 \\
3 & -2 \\
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
\end{bmatrix}
=
\begin{bmatrix}
5 \\
7 \\
\end{bmatrix}
$$

左侧的系数被称为**系数矩阵**，即

$$
\begin{bmatrix}
2 & 3 \\
3 & -2 \\
\end{bmatrix}
$$

而包含结果的矩阵称为**增广矩阵**，即

$$
\left[
\begin{array}{cc|c}
2 & 3 & 5 \\
3 & -2 & 7 \\
\end{array}
\right]
$$

对于线性方程 $Ax = b$，$A$ 是**系数矩阵** ，$(A \mid b)$ 是**增广矩阵**。

对于方程组 $Ax = 0$ 也倍称呼为 **齐次线性方程组（homogeneous linear system）**。

## 矩阵 Echelon Form 行梯形形式

如下矩阵是 4x5 的echelon form矩阵：

$$
\begin{bmatrix}
\mathbf{1} & 2 & 0 & 0 & 3 \\
0 & 0 & \mathbf{4} & 0 & 4 \\
0 & 0 & 0 & \mathbf{9} & 5 \\
0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}
$$

前侧的0被称为**先导零（leading zero）**，而非先导零的元素被称为**主元素（pivot element）** （加粗）。

## 简化行梯形形式（Reduced Row Echelon Form）

简化行梯形形式（简称RREF）是echelon form的一种特殊形式，它满足以下条件：
- 呈行梯形形式 （必须是一个 echelon form）
- 每个非零行中的首项为1（称为前导一）。
- 每列包含一个前导1，其其他所有条目均为零。

允许 0 行

如下矩阵是 4x5 的RREF矩阵：

$$
\begin{bmatrix}
\mathbf{1} & 9 & 0 & 3 & 3 \\
0 & 0 & \mathbf{1} & -3 & 4 \\
0 & 0 & 0 & \mathbf{1} & 5 \\
0 & 0 & 0 & 0 & 0 \\
\end{bmatrix}
$$

## 自由变量

对于增广矩阵 $(A \mid b)$，我们可以用高斯消元法将 A 化为行梯形形式，然后再化为简化行梯形形式。

**自由变量（free variable）** 是指在简化行梯形形式中，其列不存在主元素的变量。

$$
\begin{bmatrix}
1 & \mathbf{2} & 0 & 0 & 3 \\
0 & 0 & 1 & 0 & 4 \\
0 & 0 & 0 & 1 & \mathbf{5}
\end{bmatrix}
$$

例如列2和列4不存在主元素，因此 $x_2$ 和 $x_4$ 是自由变量。

## 无解和唯一解

如果增广矩阵的最后一列不是主元素，那么方程组没有解，例如

$$
\begin{bmatrix}
1 & 2 & 0 & 3 \\
0 & 0 & 1 & 4 \\
0 & 0 & 0 & \mathbf{5}
\end{bmatrix}
$$

其化作方程为

```
x1 + 2 x2    = 3
          x3 = 4
           0 = 5
```

对于没有自由变量的简化行梯形形式，特解（Special Solution）就是唯一解。

## 特解（Special Solution）与通解（General Solution）

而如果存在自由变量，那么解就不是唯一的，而是有无穷多个解。我们可以通过取一个特解再加上自由变量的线性组合来表示所有的解，这就是通解。

例如对于 $n$ 个变量，$f$ 个自由变量的矩阵 $M =(A \mid b)$ 来说，我们可以给所有自由变量赋予一个特殊参数，以次得到一个特殊解 $x_S$。

例如如果 $x_z$ 是自由变量，则我们可以固定其他自由变量为0，$x_z$ 为1，然后求其于齐次系统 $Ax = 0$ 的解（即 $(A 0)$），得到一个解 $x_z$。

我们为此可以得到 $f$ 个关于自由变量的齐次解，然后将其与自由变量的线性组合，即可得到通解。

即

$$
x = x_S + \lambda_1 x_1 + \lambda_2 x_2 + \cdots + \lambda_f x_f, \lambda_{i...f} \in \mathbb{R}
$$


用集合形式表示为

$$
\left\{
x\in \mathbb{R}^n : x = x_S + \lambda_1 x_1 + \lambda_2 x_2 + \cdots + \lambda_f x_f, \lambda_{i...f} \in \mathbb{R}
\right\}
$$

$$
\left\{
x\in \mathbb{R}^n : x = x_S + \sum_{x_i \text{ is free}}{\lambda_i x_i}, \lambda \in \mathbb{R}
\right\}
$$

用span形式表示为

$$
x = x_S + \text{span}\{x_1, x_2, \cdots, x_f\}
$$

我们可以把特解理解为在目标空间中的一个点，而自由变量的齐次解理解为一个向量空间，而通解就是特解与向量空间的线性组合。  
而使用齐次解是因为我们可以把 $Ax=b$ 中的 $b$ 看作是 $Ax=0$ 解对于空间的平移，而平移并不会影响新生成的（超）平面等的方向大小等，只会影响其的位置，而位置的变化可以通过特解来表示。

简单来说就是；
1. 寻找 $Ax = b$ 的特解 $x_S$
2. 寻找所有自由变量的齐次解 $Ax = 0$
3. 将特解与自由变量的齐次解的线性组合，即可得到通解。


## 平凡解（Trivial Solution）

对于齐次系统 $Ax = 0$，其一定包含一个解 $x=0$，称为平凡解。
而除了平凡解之外（非零解）的解称为非平凡解。

## Rank

其实，矩阵的秩（Rank）就是矩阵的行梯形形式中非零行的数量。  
也就是Reduced Row Echelon Form先导1的个数。

对于一个 $m \times n$ 的矩阵 $M$（m行n列），$r = \text{Rank}(M)$

- 列满秩：n个列向量线性无关

## Elementary Row Transformation / 初等行变换

初等行变换是指对矩阵进行的以下三种操作：
1. （对换变换）交换两行
2. （倍乘变换）一行乘以一个非零标量
3. （倍加变换）用一个非零常数乘以一行，然后加到另一行