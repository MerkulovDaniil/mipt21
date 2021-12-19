---
title: 💀 Домашка
nav_order: 3
---

# Домашнее задание 1.

The file should be sent in the `.pdf` format created via $$\LaTeX$$ or [typora](<https://typora.io/>) or `print to pdf` option from the colab\jupyter notebook.

**Deadline:** 16 October, 15:59 (Moscow time).

## Matrix calculus

1. Find $$\nabla f(x)$$, if $$f(x) = \dfrac{1}{2} \|Ax - b\|_2^2 , x \in \mathbb{R}^p$$.
1. Find $$\nabla f(x)$$, if $$f(x) = \langle x, x\rangle^{\langle x, x\rangle}, x \in \mathbb{R}^n\setminus\{0\}$$.
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
1. Prove, that the set $S \subseteq \mathbb{R}^n$ is convex if and only if $(\alpha + \beta)S = \alpha S + \beta S$ for all non-negative $\alpha$ and $\beta$.
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
	
	Prove, that $$D(p,q) \geq 0 \; \forall p,q \in \mathbb{R}^n_{++}$$ и $$D(p,q) = 0 \leftrightarrow p = q$$  
	
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
1.  Is the function returning the arithmetic mean of vector coordinates is a convex one: $$a(x) = \frac{1}{n}\sum\limits_{i=1}^n x_i$$, what about geometric mean: $$g(x) = \prod\limits_{i=1}^n \left(x_i \right)^{1/n}$$?
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

# Домашнее задание 2

The file should be sent in the `.pdf` format created via $$\LaTeX$$ or [typora](<https://typora.io/>) or `print to pdf` option from the colab\jupyter notebook. The only handwritten part, that could be included in the solution are the figures and illustrations.

**Deadline:** 05 December, 15:59 (Moscow time).


## General optimization problems

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & Ax = b
	\end{split}
	$$

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = 1, \\
	& x \succeq 0 
	\end{split}
	$$

	This problem can be considered as a simplest portfolio optimization problem.

1. Give an explicit solution of the following LP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & 1^\top x = \alpha, \\
	& 0 \preceq x \preceq 1,
	\end{split}
	$$

	where $$\alpha$$ is an integer between $$0$$ and $$n$$. What happens if $$\alpha$$ is not an integer (but satisfies $$0 \leq \alpha \leq n$$)? What if we change the equality to an inequality $$1^\top x \leq \alpha$$?

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & x^\top A x \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, c \neq 0$$. What is the solution if the problem is not convex $$(A \notin \mathbb{S}^n_{++})$$ (Hint: consider eigendecomposition of the matrix: $$A = Q \mathbf{diag}(\lambda)Q^\top = \sum\limits_{i=1}^n \lambda_i q_i q_i^\top$$ and different cases of $$\lambda >0, \lambda=0, \lambda<0$$)?

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& c^\top x \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & (x - x_c)^\top A (x - x_c) \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, c \neq 0, x_c \in \mathbb{R}^n$$.

1. Give an explicit solution of the following QP.
	
	$$
	\begin{split}
	& x^\top Bx \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & x^\top A x \leq 1,
	\end{split}
	$$

	where $$A \in \mathbb{S}^n_{++}, B \in \mathbb{S}^n_{+}$$.

1.  Consider the equality constrained least-squares problem
	
	$$
	\begin{split}
	& \|Ax - b\|_2^2 \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & Cx = d,
	\end{split}
	$$

	where $$A \in \mathbb{R}^{m \times n}$$ with $$\mathbf{rank }A = n$$, and $$C \in \mathbb{R}^{k \times n}$$ with $$\mathbf{rank }C = k$$. Give the KKT conditions, and derive expressions for the primal solution $$x^*$$ and the dual solution $$\lambda^*$$.

1. Derive the KKT conditions for the problem
	
	$$
	\begin{split}
	& \mathbf{tr \;}X - \log\text{det }X \to \min\limits_{X \in \mathbb{S}^n_{++} }\\
	\text{s.t. } & Xs = y,
	\end{split}
	$$

	where $$y \in \mathbb{R}^n$$ and $$s \in \mathbb{R}^n$$ are given with $$y^\top s = 1$$. Verify that the optimal solution is given by

	$$
	X^* = I + yy^\top - \dfrac{1}{s^\top s}ss^\top
	$$

1.  **Supporting hyperplane interpretation of KKT conditions**. Consider a **convex** problem with no equality constraints
	
	$$
	\begin{split}
	& f_0(x) \to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & f_i(x) \leq 0, \quad i = [1,m]
	\end{split}
	$$

	Assume, that $$\exists x^* \in \mathbb{R}^n, \mu^* \in \mathbb{R}^m$$ satisfy the KKT conditions
	
	$$
	\begin{split}
    & \nabla_x L (x^*, \mu^*) = \nabla f_0(x^*) + \sum\limits_{i=1}^m\mu_i^*\nabla f_i(x^*) = 0 \\
    & \mu^*_i \geq 0, \quad i = [1,m] \\
    & \mu^*_i f_i(x^*) = 0, \quad i = [1,m]\\
    & f_i(x^*) \leq 0, \quad i = [1,m]
	\end{split}
	$$

	Show that

	$$
	\nabla f_0(x^*)^\top (x - x^*) \geq 0
	$$

	for all feasible $$x$$. In other words the KKT conditions imply the simple optimality criterion or $$\nabla f_0(x^*)$$ defines a supporting hyperplane to the feasible set at $$x^*$$.

## Duality

1.  **Fenchel + Lagrange = ♥.** Express the dual problem of
	
	$$
	\begin{split}
	& c^\top x\to \min\limits_{x \in \mathbb{R}^n }\\
	\text{s.t. } & f(x) \leq 0
	\end{split}
	$$

	with $$c \neq 0$$, in terms of the conjugate function $$f^*$$. Explain why the problem you give is convex. We do not assume $$f$$ is convex.

1. **Minimum volume covering ellipsoid.** Let we have the primal problem:
	
	$$
	\begin{split}
	& \ln \text{det} X^{-1} \to \min\limits_{X \in \mathbb{S}^{n}_{++} }\\
	\text{s.t. } & a_i^\top X a_i \leq 1 , i = 1, \ldots, m
	\end{split}
	$$

	1. Find Lagrangian of the primal problem
	1. Find the dual function
	1. Write down the dual problem
	1. Check whether problem holds strong duality or not
	1. Write down the solution of the dual problem
	
1. **A penalty method for equality constraints.** We consider the problem of minimization
	$$
	\begin{split}
	& f_0(x) \to \min\limits_{x \in \mathbb{R}^{n} }\\
	\text{s.t. } & Ax = b,
	\end{split}
	$$
	
	where $$f_0(x): \mathbb{R}^n \to\mathbb{R} $$ is convex and differentiable, and $$A \in \mathbb{R}^{m \times n}$$ with $$\mathbf{rank }A = m$$. In a quadratic penalty method, we form an auxiliary function

	$$
	\phi(x) = f_0(x) + \alpha \|Ax - b\|_2^2,
	$$
	
	where $$\alpha > 0$$ is a parameter. This auxiliary function consists of the objective plus the penalty term $$\alpha \|Ax - b\|_2^2$$. The idea is that a minimizer of the auxiliary function, $$\tilde{x}$$, should be an approximate solution of the original problem. Intuition suggests that the larger the penalty weight $$\alpha$$, the better the approximation $$\tilde{x}$$ to a solution of the original problem. Suppose $$\tilde{x}$$ is a minimizer of $$\phi(x)$$. Show how to find, from $$\tilde{x}$$, a dual feasible point for the original problem. Find the corresponding lower bound on the optimal value of the original problem.
	
1. **Analytic centering.** Derive a dual problem for
	
	$$
	-\sum_{i=1}^m \log (b_i - a_i^\top x) \to \min\limits_{x \in \mathbb{R}^{n} }
	$$

	with domain $$\{x \mid a^\top_i x < b_i , i = [1,m]\}$$. 
	
	First introduce new variables $$y_i$$ and equality constraints $$y_i = b_i − a^\top_i x$$. (The solution of this problem is called the analytic center of the linear inequalities $$a^\top_i x \leq b_i ,i = [1,m]$$.  Analytic centers have geometric applications, and play an important role in barrier methods.) 

## Applications

1. **📱🎧💻 Covers manufacturing.** Random Corp is producing covers for following products: 
	* 📱 phones
	* 🎧 headphones
	* 💻 laptops

	The company’s production facilities are such that if we devote the entire production to headphones covers, we can produce $$6000$$ of them in one day. If we devote the entire production to phone covers or laptop covers, we can produce $$5000$$ or $$3000$$ of them in one day. 

	The production schedule is one week ($$5$$ working days), and the week’s production must be stored before distribution. Storing $$1000$$ headphones covers (packaging included) takes up $$50$$ cubic feet of space. Storing $$1000$$ phone covers (packaging included) takes up $$60$$ cubic feet of space, and storing $$1000$$ laptop covers (packaging included) takes up $$220$$ cubic feet of space. The total storage space available is $$6000$$ cubic feet. 

	Due to commercial agreements with Random Corp has to deliver at least $$5000$$ headphones covers and $$4000$$ laptop covers per week in order to strengthen the product’s diffusion. 

	The marketing department estimates that the weekly demand for headphones covers, phone, and laptop covers does not exceed $$10000$$ and $$15000$$, and $$8000$$ units, therefore the company does not want to produce more than these amounts for headphones, phone, and laptop covers. 

	Finally, the net profit per each headphones cover, phone cover, and laptop cover is $$\$5$$, $$\$7$$, and $$\$12$$, respectively.

	The aim is to determine a weekly production schedule that maximizes the total net profit.

	Write a correct Linear Programming formulation for the problem.	Use following variables:

	* $$y_1$$ = number of headphones covers produced over the week,  
	* $$y_2$$ = number of phone covers produced over the week,  
	* $$y_3$$ = number of laptop covers produced over the week.  

	Find the solution to the problem using `PyOMO.` Take a look at the example , described in the class [🐍 code](https://colab.research.google.com/github/MerkulovDaniil/sber219/blob/main/notebooks/8_01.ipynb){: .btn}.

	```python
	!pip install pyomo --quiet
	! sudo apt-get install glpk-utils --quiet  # GLPK
	! sudo apt-get install coinor-cbc --quiet  # CoinOR
	```

1. **📼 Optimal watching TED talks** In this task you are to formulate LP problem for selecting TED talks for watching. Take a look at the example , described in the class  [🐍 code](https://colab.research.google.com/github/MerkulovDaniil/sber219/blob/main/notebooks/8_1.ipynb){: .btn}.
	
	```python
	!pip install pulp
	
	# Commented out IPython magic to ensure Python compatibility.
	# %matplotlib inline
	 
	import pulp
	import numpy as np
	import pandas as pd
	import re
	import matplotlib.pyplot as plt
	from IPython.display import Image
	
	# Download the dataset
	
	# Read the dataset into pandas dataframe, convert duration from seconds to minutes
	ted = pd.read_csv('https://raw.githubusercontent.com/MerkulovDaniil/optim/master/assets/Notebooks/ted_main.csv', encoding='ISO-8859-1')
	ted['duration'] = ted['duration'] / 60
	ted = ted.round({'duration': 1})
	
	# Select subset of columns & rows (if required)
	# data = ted.sample(n=1000) # 'n' can be changed as required
	data = ted
	selected_cols = ['name', 'event', 'duration', 'views']
	data.reset_index(inplace=True)
	data.head()
	
	# create LP object,
	# set up as a maximization problem --> since we want to maximize the number of TED talks to watch
	prob = pulp.LpProblem('WatchingTEDTalks', pulp.LpMaximize)
	
	# create decision - yes or no to watch the talk?
	decision_variables = []
	for rownum, row in data.iterrows():
	    variable = str('x' + str(row['index']))
	    variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1) # make variable binary
	    decision_variables.append(variable)
	    
	print('Total number of decision variables: ' + str(len(decision_variables)))
	
	###
	### YOUR CODE HERE
	###
	
	
	# Be careful, the output will be huge
	# print(prob)
	prob.writeLP('WatchingTEDTalks.lp')
	print('🤔 The problem has successfully formulated')
	
	optimization_result = prob.solve()
	
	assert optimization_result == pulp.LpStatusOptimal
	print('Status:', pulp.LpStatus[prob.status])
	print('Optimal Solution to the problem: ', pulp.value(prob.objective))
	print('Individual decision variables: ')
	
	for v in prob.variables():
	    if v.varValue > 0:
	        print(v.name, '=', v.varValue)
	
	# reorder results
	variable_name = []
	variable_value = []
	
	for v in prob.variables():
	    variable_name.append(v.name)
	    variable_value.append(v.varValue)
	    
	df = pd.DataFrame({'index': variable_name, 'value': variable_value})
	for rownum, row in df.iterrows():
	    value = re.findall(r'(\d+)', row['index'])
	    df.loc[rownum, 'index'] = int(value[0])
	    
	# df = df.sort_index(by = 'index')
	df = df.sort_values(by = 'index')
	result = pd.merge(data, df, on = 'index')
	result = result[result['value'] == 1].sort_values(by = 'views', ascending = False)
	selected_cols_final = ['name', 'event', 'duration', 'views']
	final_set_of_talks_to_watch = result[selected_cols_final]
	
	from IPython.display import display, HTML
	display(HTML(final_set_of_talks_to_watch.to_html(index=False)))
	```

	1. Your task is to choose linear objective function and justify your choice. 
	1. Then select at least $3$ various non-trivial linear constraints on the solution. (For example, bound the average length of the selected videos or etc.)
	1. Solve the problem with `pulp` library using code above.

1. **Risk budget allocation.** [source](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook_extra_exercises.pdf). Suppose an amount $$x_i>0$$ is invested in $$n$$ assets, labeled $$i=1,..., n$$, with asset return covariance matrix $$\Sigma \in \mathcal{S}_{++}^n$$. We define the *risk* of the investments as the standard deviation of the total return, $$R(x) = (x^\top\Sigma x)^{1/2}$$.

	We define the (relative) *risk contribution* of asset $$i$$ (in the portfolio $$x$$) as

	$$
	\rho_i = \frac{\partial \log R(x)}{\partial \log x_i} = \frac{\partial R(x)}{R(x)} \frac{x_i}{\partial x_i}, \quad i=1, \ldots, n.
	$$

	Why is the logarithm here?! Because it reflects fraction of relative change $$x$$ (say, per 1%). Take a look at [easticity definition at wiki](https://en.wikipedia.org/wiki/Elasticity_(economics)). Thus $$\rho_i$$ gives the fractional increase in risk per fractional increase
	in investment $$i$$. We can express the risk contributions as

	$$
	\rho_i = \frac{x_i (\Sigma x)_i} {x^\top\Sigma x}, \quad i=1, \ldots, n,
	$$

	from which we see that $$\sum_{i=1}^n \rho_i = 1$$. For general $$x$$, we can have $$\rho_i <0$$, which means that a small increase in investment $i$ decreases the risk. Desirable investment choices have $$\rho_i>0$$, in which case we can interpret $$\rho_i$$ as the fraction of the total risk contributed by the investment in asset $$i$$. Note that the risk contributions are homogeneous, i.e., scaling $$x$$ by a positive constant does not affect $$\rho_i$$.

	**Problem statement**

	In the *risk budget allocation problem*, we are given $$\Sigma$$ and a set of desired risk contributions $$\rho_i^\mathrm{des}>0$$ with $$\bf{1}^\top \rho^\mathrm{des}=1$$; the goal is to find an investment mix $x\succ 0$, $\bf{1}^\top x =1$, with these risk contributions.

	When $$\rho^\mathrm{des} = (1/n)\bf{1}$$, the problem is to find an investment mix that achieves so-called *risk parity*.

	1. Explain how to solve the risk budget allocation problem using convex optimization.

		*Hint.* Minimize $$(1/2)x^\top\Sigma x - \sum_{i=1}^n \rho_i^\mathrm{des} \log x_i$$.
	1. Find the investment mix that achieves risk parity for the return covariance matrix $$\Sigma$$ below.

	```python
	import numpy as np
	import cvxpy as cp
	Sigma = np.array(np.matrix("""6.1  2.9  -0.8  0.1;
	                     2.9  4.3  -0.3  0.9;
	                    -0.8 -0.3   1.2 -0.7;
	                     0.1  0.9  -0.7  2.3"""))
	rho = np.ones(4)/4
	```

1. **💲 📈📉 Как с деньгами обстоит вопрос?** In this task you are to select your own portfolio and formulate your favorite optimization problem to solve it with cvxpy. Take a look at the example, described in the class  [🐍 code](https://colab.research.google.com/github/MerkulovDaniil/sber219/blob/main/notebooks/4_1.ipynb){: .btn}.

	![](https://pbs.twimg.com/media/Ef-YalfXYAAUDaf.jpg)

	```python
	!pip install yfinance --quiet

	import datetime
	import matplotlib.pyplot as plt
	import pandas
	import numpy as np
	from pandas_datareader import data as pdr
	import yfinance as yfin
	yfin.pdr_override()

	### YOUR STOCKS HERE
	#stocks = ['GOOGL', 'SPY', 'AAPL', 'TSLA', 'MSFT']
	stocks = []
	df = pdr.get_data_yahoo(stocks, start="2021-01-01")

	# Adjusted price
	for stock in stocks:
	    plt.plot(df['Adj Close'][stock], label=stock)
	plt.xlabel('date')
	plt.ylabel('Adjusted price')
	plt.legend()
	plt.show()

	# Fractional return
	frac_return = {}
	for stock in stocks:
	    frac_return[stock] = [(price - df['Adj Close'][stock][0])/df['Adj Close'][stock][0] for price in df['Adj Close'][stock]]
	    plt.plot(frac_return[stock], label=stock)
	plt.xlabel('date')
	plt.ylabel('Fractional return')
	plt.legend()
	plt.show()
	```


