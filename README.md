# 1D-diffusion-equation
A solver for 1D diffusion equation using both analytical and numerical methods

## One-dimentional diffusion equation:
$$\displaystyle\frac{\partial c}{\partial t} = D \displaystyle\frac{\partial^2 c}{\partial x^2},$$

where $$D$$ is the diffusion coefficient, $$t$$ is time, and $$x$$ is the spatial coordinate, $$c$$ is the concentration, which is function of $$D$$, $$t$$, $$x$$ and total mass $$M$$.

## Analytical solution
The analytical solution with and without boundaries of the diffusion equation can be obtained using the fundamental solution and the source image method, respectively:

$$c(x,t)=\frac{M}{\sqrt{4\pi Dt}}\exp\left(-\frac{x^2}{4Dt}\right),$$

$$c(x,t)=\frac{M}{\sqrt{4\pi Dt}} \sum_{n=-\infty}^{\infty} \exp\left[-\frac{(x+2nL)^2}{4Dt}\right],$$

where $$L$$ is the distance between the source and the boundary. The fundamental solution is used for cases without boundaries, while the source image method is used for cases with boundaries. The analytical solutions provide a theoretical basis for understanding the diffusion process and serve as a benchmark for numerical solutions.

## Numerical solution
### Explicit scheme
The discretization of the explicit algorithm is even simple:

$$c_{i,j+1} = c_{i,j} + \alpha(c_{i+1,j}-2c_{i,j}+c_{i-1,j}).$$

The time step is chosen as 0.0001, and the spatial step is chosen as 0.01. The Courant number is calculated as:

$$\text{CFL} = \displaystyle\frac{D \Delta t}{\Delta x^2} = 0.1 < 0.5.$$

A smaller Courant number can ensure the numerical stability of the computation.

### Implicit scheme
Using Crank-Nicolson scheme, the diffusion equation can be discretized as:

$$\displaystyle\frac{c_{i,j+1} - c_{i,j}}{\Delta t} = D \displaystyle\frac{\displaystyle\frac{1}{2} \left[c_{i+1,j}-2c_{i,j}+c_{i-1,j}+c_{i+1,j+1}-2c_{i,j+1}-c_{i-1,j-1}\right]}{\Delta x^2}.$$

Introduce $$\alpha$$ to make some substitutions: $$\alpha = \displaystyle\frac{D \Delta t}{\Delta x^2}$$. Then the equation can be rewritten as:

$$-\alpha c_{i-1,j+1}+2(1+\alpha)c_{i,j+1}-\alpha c_{i+1,j+1} = \alpha c_{i-1,j}+2(1-\alpha)c_{i,j}+\alpha c_{i+1,j}.$$

Neumann conditions is applied to control the boundary, ensuring that the gradient at the boundary is equal to 0:

$$
    \begin{matrix}
        \displaystyle\frac{\partial c_{1,j}}{\partial x} = 0 \rightarrow \displaystyle\frac{c_{0,j}-c_{1,j}}{\Delta x} = 0 \rightarrow c_{0,j} = c_{1,j},\\
        \displaystyle\frac{\partial c_{N-1,j}}{\partial x} = 0 \rightarrow \displaystyle\frac{c_{N,j}-c_{N-1,j}}{\Delta x} = 0 \rightarrow c_{N-1,j} = c_{N,j}.
    \end{matrix}
$$

Converting the equation system into matrix form: $$AC=B$$, where:

$$
A = \left[\begin{matrix}
        2+\alpha&  -\alpha&   0&   0& 0 & & & 0\\
        -\alpha&   2(1+\alpha)&  -\alpha&   0&  0 & \dots& & 0\\
        0&  -\alpha&   2(1+\alpha)&  -\alpha&  0 & & & 0\\
        \vdots&  &  &  \ddots&  &  & & \vdots\\
        0&  0&  & 0& -\alpha&   2(1+\alpha)&  -\alpha &0\\
        0&  0&  \dots& & 0& -\alpha&   2(1+\alpha)&  -\alpha\\
        0&  0&   & & & & -\alpha& 2+\alpha\\
    \end{matrix}\right],
$$

$$
C = \left[\begin{array}{l}
            c_{1,j+1}, c_{2,j+1}, c_{3,j+1}, \dots, c_{N-3,j+1}, c_{N-2,j+1}, c_{N-1,j+1}\\
        \end{array}\right]^T,
$$

$$
B = \left[\begin{matrix}
        (2-\alpha)c_{1,j}+\alpha c_{2,j}\\
        \alpha c_{1,j}+2(1-\alpha)c_{2,j}+\alpha c_{3,j}\\
        \alpha c_{2,j}+2(1-\alpha)c_{3,j}+\alpha c_{4,j}\\
        \vdots\\
        \alpha c_{N-4,j}+2(1-\alpha)c_{N-3,j}+\alpha c_{N-2,j}\\
        \alpha c_{N-3,j}+2(1-\alpha)c_{N-2,j}+\alpha c_{N-1,j}\\
        \alpha c_{N-2,j}+(2-\alpha)c_{N-1,j}
    \end{matrix}\right]
$$

The treatment of boundaries and grid partitioning is consistent with the approach mentioned in the explicit algorithm.

## Results

Numerical solutions conmpared with analtical solutions:

### Explicit scheme ###
![image](https://github.com/ZimoJupiter/1D-diffusion-equation/blob/main/Figures/Explicit.png)

### Imlicit scheme ###
![image](https://github.com/ZimoJupiter/1D-diffusion-equation/blob/main/Figures/Implicit.png)
