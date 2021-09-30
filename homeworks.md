---
title: 💀 Домашка
nav_order: 3
---

# Домашнее задание 1.

The file should be sent in the `.pdf` format created via $$\LaTeX$$ or [typora](<https://typora.io/>) or `print to pdf` option from the colab\jupyter notebook.

**Deadline:** 16 October, 15:59 (Moscow time).

## Matrix calculus

1. Find $$\nabla f(x)$$, if $$f(x) = \dfrac{1}{2} \|Ax - b\|_2^2 , x \in \mathbb{R}^p$$.
1. Find $$\nabla f(X)$$, if $$f(X) = \langle x, x\rangle^{\langle x, x\rangle}, x \in \mathbb{R}^n\setminus\{0\}$$.
1. Calculate the Frobenious norm derivative: $$\dfrac{\partial}{\partial X}\|X\|_F^2$$
1. Calculate the first and the second derivative of the following function $$f : S \to \mathbb{R}$$
	$$
	f(t) = \text{det}(A − tI_n),
	$$
	where $$A \in \mathbb{R}^{n \times n}, S := \{t \in \mathbb{R} : \text{det}(A − tI_n) \neq 0\}	$$.
1. Implement analytical expression of the gradient and hessian of the following functions:
	a. $$f(x) = \dfrac{1}{2}x^TAx + b^Tx + c$$
	b. $$f(x) = \ln \left( 1 + \exp\langle a,x\rangle\right)$$
	c. $$f(x) = \dfrac{1}{2} \|Ax - b\|^2_2$$

	and compare the analytical answers with those, which obtained with any automatic differentiation framework (autograd\jax\pytorch\tensorflow). Manuals: [Jax autograd manual](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html), [general manual](https://colab.research.google.com/github/MerkulovDaniil/optim/blob/master/assets/Notebooks/Autograd.ipynb).

```python
import numpy as np

n = 10
A = np.random.rand((n,n))
b = np.random.rand(n)
c = np.random.rand(n)

def f(x):
    # Your code here
    return 0

def analytical_df(x):
    # Your code here
    return np.zeros(n)

def analytical_ddf(x):
    # Your code here
    return np.zeros((n,n))

def autograd_df(x):
    # Your code here
    return np.zeros(n)

def autograd_ddf(x):
    # Your code here
    return np.zeros((n,n))

x_test = np.random.rand(n)

print(f'Analytical and autograd implementations of the gradients are close: {np.allclose(analytical_df(x_test), autograd_df(x_test))}')
print(f'Analytical and autograd implementations of the hessians are close: {np.allclose(analytical_ddf(x_test), autograd_ddf(x_test))}')

```

## Convex sets

1. Prove that the set of square symmetric positive definite matrices is convex.
1. Show, that $$ \mathbf{conv}\{xx^\top: x \in \mathbb{R}^n, \|x\| = 1\} = \{A \in \mathbb{S}^n_+: \text{tr}(A) = 1\}$$.
1. Show that the hyperbolic set of $$ \{x \in \mathbb{R}^n_+ | \prod\limits_{i=1}^n x_i \geq 1 \} $$ is convex. 
Hint: For $$0 \leq \theta \leq 1$$ it is valid, that $$a^\theta b^{1 - \theta} \leq \theta a + (1-\theta)b$$ with non-negative $$a,b$$.
1. Prove, that the set $S \subseteq \mathbb{R}^n$ is convex if and only if $(\alpha + \beta)S = \alpha S + \beta S$ for all non-negative $\alpha$ and $\beta$
1. Let $$x \in \mathbb{R}$$ is a random variable with a given probability distribution of $$\mathbb{P}(x = a_i) = p_i$$, where $$i = 1, \ldots, n$$, and $$a_1 < \ldots < a_n$$. It is said that the probability vector of outcomes of $$p \in \mathbb{R}^n$$ belongs to the probabilistic simplex, i.e. $$P = \{ p \mid \mathbf{1}^Tp = 1, p \succeq 0 \} = \{ p \mid p_1 + \ldots + p_n = 1, p_i \ge 0 \}$$. 
    Determine if the following sets of $$p$$ are convex:
    
	1. \$$\mathbb{P}(x > \alpha) \le \beta$$
	1. \$$\mathbb{E} \vert x^{201}\vert \le \alpha \mathbb{E}\vert x \vert$$
	1. \$$\mathbb{E} \vert x^{2}\vert \ge \alpha $$
	1. \$$\mathbb{V}x \ge \alpha$$

## Convex functions

1. Prove, that function $$f(X) = \mathbf{tr}(X^{-1}), X \in S^n_{++}$$ is convex, while $$g(X) = (\det X)^{1/n}, X \in S^n_{++}$$ is concave.
1. Kullback–Leibler divergence between $$p,q \in \mathbb{R}^n_{++}$$ is:
	
	$$
	D(p,q) = \sum\limits_{i=1}^n (p_i \log(p_i/q_i) - p_i + q_i)
	$$
	
	Prove, that $$D(p,q) \geq 0 \forall p,q \in \mathbb{R}^n_{++}$$ и $$D(p,q) = 0 \leftrightarrow p = q$$  
	
	Hint: 
	$$
	D(p,q) = f(p) - f(q) - \nabla f(q)^T(p-q), \;\;\;\; f(p) = \sum\limits_{i=1}^n p_i \log p_i
	$$
1. Let $$x$$ be a real variable with the values $$a_1 < a_2 < \ldots < a_n$$ with probabilities $$\mathbb{P}(x = a_i) = p_i$$. Derive the convexity or concavity of the following functions from $$p$$ on the set of $$\left\{p \mid \sum\limits_{i=1}^n p_i = 1, p_i \ge 0 \right\}$$  
	* \$$\mathbb{E}x$$
	* \$$\mathbb{P}\{x \ge \alpha\}$$
	* \$$\mathbb{P}\{\alpha \le x \le \beta\}$$
	* \$$\sum\limits_{i=1}^n p_i \log p_i$$
	* \$$\mathbb{V}x = \mathbb{E}(x - \mathbb{E}x)^2$$
	* \$$\mathbf{quartile}(x) = {\operatorname{inf}}\left\{ \beta \mid \mathbb{P}\{x \le \beta\} \ge 0.25 \right\}$$ 
1.  Is the function returning the arithmetic mean of vector coordinates is a convex one: : $$a(x) = \frac{1}{n}\sum\limits_{i=1}^n x_i$$, what about geometric mean: $$g(x) = \prod\limits_{i=1}^n \left(x_i \right)^{1/n}$$?
1. Is $$f(x) = -x \ln x - (1-x) \ln (1-x)$$ convex?
1. Let $$f: \mathbb{R}^n \to \mathbb{R}$$ be the following function:
    $$
    f(x) = \sum\limits_{i=1}^k x_{\lfloor i \rfloor},
    $$
    where $$1 \leq k \leq n$$, while the symbol $$x_{\lfloor i \rfloor}$$ stands for the $$i$$-th component of sorted ($$x_{\lfloor 1 \rfloor}$$ - maximum component of $$x$$ and $$x_{\lfloor n \rfloor}$$ - minimum component of $$x$$) vector of $$x$$. Show, that $$f$$ is a convex function.

    
## Conjugate sets

1. Let $$\mathbb{A}_n$$ be the set of all $$n$$ dimensional antisymmetric matrices. Show that $$\left( \mathbb{A}_n\right)^* = \mathbb{S}_n$$. 
1. Find the conjugate set to the ellipsoid: 
    
    $$
     S = \left\{ x \in \mathbb{R}^n \mid \sum\limits_{i = 1}^n a_i^2 x_i^2 \le \varepsilon^2 \right\}
    $$
1. Find the sets $$S^{*}, S^{**}, S^{***}$$, if 
    
    $$
    S = \{ x \in \mathbb{R}^2 \mid x_1 + x_2 \ge -1, \;\; 2x_1 + x_2 \ge 1, \;\; -2x_1 + x_2 \ge 2\}
    $$
1. Find the conjugate cone for the exponential cone:
    
    $$
    K = \{(x, y, z) \mid y > 0, y e^{x/y} \leq z\}
    $$
1. Prove, that $$B_p$$ and $$B_{p_*}$$ are inter-conjugate, i.e. $$(B_p)^* = B_{p_*}, (B_{p_*})^* = B_p$$, where $$B_p$$ is the unit ball (w.r.t. $$p$$ - norm) and $$p, p_*$$ are conjugated, i.e. $$p^{-1} + p^{-1}_* = 1$$. You can assume, that $$p_* = \infty$$ if $$p = 1$$ and vice versa.

## Conjugate function

1. Find $$f^*(y)$$, if $$f(x) = -\dfrac{1}{x}, \;\; x\in \mathbb{R}_{++}$$
1. Prove, that if $$f(x_1, x_2) = g_1(x_1) + g_2(x_2)$$, then $$f^*(y_1, y_2) = g_1^*(y_1) + g_2^*(y_2)$$
1. Find $$f^*(y)$$, if $$f(x) = \log \left( \sum\limits_{i=1}^n e^{x_i} \right)$$
1. Prove, that if $$f(x) = \alpha g(x)$$, then $$f^*(y) = \alpha g^*(y/\alpha)$$
1. Find $$f^*(Y)$$, if $$f(X) = - \ln \det X, X \in \mathbb{S}^n_{++}$$
1. Prove, that if $$f(x) = \inf\limits_{u+v = x} (g(u) + h(v))$$, then $$f^*(y) = g^*(y) + h^*(y)$$

## Subgradient and subdifferential

1. Prove, that $$x_0$$ - is the minimum point of a convex function $$f(x)$$ if and only if $$0 \in \partial f(x_0)$$
1. Find $$\partial f(x)$$, if $$f(x) = \text{ReLU}(x) = \max \{0, x\}$$
1. Find $$\partial f(x)$$, if $$f(x) = \|x\|_p$$ при $$p = 1,2, \infty$$
1. Find $$\partial f(x)$$, if $$f(x) = \|Ax - b\|_1$$
1. Find $$\partial f(x)$$, if $$f(x) = e^{\|x\|}$$
1. Find $$\partial f(x)$$, if $$f(x) = \max\limits_i\left\{ \langle a_i, x\rangle + b_i \right\}, a_i \in \mathbb{R}^n, b_i \in \mathbb{R}, i = 1, \ldots, m$$
