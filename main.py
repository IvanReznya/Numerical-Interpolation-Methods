import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("images", exist_ok=True)


def Barycentric(x_nodes ,y_nodes, x):
    N = len(x_nodes)
    w = np.ones(N)
    for i in range(N):
        for j in range(N):
            if i !=j:
                w[i] /= (x_nodes[i] - x_nodes[j])

    for i in range(N):
        if abs(x - x_nodes[i]) < 1e-16:
            return y_nodes[i]

    res_1 = 0
    res_2 = 0

    for i in range(N):
        res_1 += w[i]*y_nodes[i]/(x - x_nodes[i])
        res_2 += w[i]/(x - x_nodes[i])
    return res_1/res_2


def Lagrange(x_nodes, y_nodes):
    x_plot = np.linspace(x_nodes[0], x_nodes[len(x_nodes)-1], 100)
    poly = [Barycentric(x_nodes, y_nodes, x) for x in x_plot]
    plt.plot(x_plot, poly)
    plt.scatter(x_nodes, y_nodes, color = 'red')
    plt.axvline(x = 0, color = 'black')
    plt.axhline(y = 0, color = 'black')
    plt.xlabel('x')
    plt.ylabel('L(x)')
    plt.grid()
    plt.xlim(min(x_nodes) - 50, max(x_nodes) + 50)
    plt.ylim(min(y_nodes) - 100, max(y_nodes) + 100)


def Spline(x_nodes, y_nodes, second_derivative_right):

    N = len(x_nodes) - 1
    h = np.array([x_nodes[i+1] - x_nodes[i] for i in range(N)])
    d = np.array([(y_nodes[i+1] - y_nodes[i])/h[i] for i in range(N)])
    u = np.array([6*(d[i] - d[i-1]) for i in range(1, N)])
    u[N-2] = u[N-2] - h[N-1]*second_derivative_right

    A = np.zeros((N - 1, N - 1))
    for i in range(N - 1):
        if i == 0:
            A[i, i] = 3*h[0] + 2*h[1] #В силу краевого условия (iv) на левом конце
            if N - 1 > 1:
                A[i, i + 1] = h[i + 1]
        elif i == N - 2:
            A[i, i - 1] = h[i]
            A[i, i] = 2 * (h[i] + h[i + 1])
        else:
            A[i, i - 1] = h[i]
            A[i, i] = 2 * (h[i] + h[i + 1])
            A[i, i + 1] = h[i + 1]

    m_inner = np.linalg.solve(A, u)
    m = np.zeros(N + 1)
    m[0] = m_inner[0]
    m[1:N] = m_inner
    m[N] = second_derivative_right


    S = np.zeros((N, 4))
    for i in range(N):
        S[i, 0] = y_nodes[i]
        S[i, 1] = d[i] - (h[i] / 6) * (2 * m[i] + m[i+1])
        S[i, 2] = m[i]/2
        S[i, 3] = (1/(6*h[i]))*(m[i+1] - m[i])

    return S


def Parametric_spline(x_nodes, y_nodes, second_derivative_right):

    N = len(x_nodes) - 1
    s = np.array([np.sqrt((x_nodes[i] - x_nodes[i+1])**2 + (y_nodes[i] - y_nodes[i+1])**2) for i in range(N)])
    t = np.zeros(N+1)
    for i in range(1, N+1):
        t[i] = t[i-1] + s[i-1]

    S_1 = Spline(t, x_nodes, second_derivative_right)
    S_2 = Spline(t, y_nodes, second_derivative_right)

    return t, S_1, S_2


def Plot_spline(x_nodes, y_nodes, second_derivative_right):

    N = len(x_nodes) - 1
    S = Spline(x_nodes, y_nodes, second_derivative_right)

    for i in range(N):
        x_segment = np.linspace(x_nodes[i], x_nodes[i + 1], 100)
        plt.plot(x_segment, S[i, 0] + S[i, 1] * (x_segment - x_nodes[i]) + S[i, 2] * (x_segment - x_nodes[i]) ** 2 + S[i, 3] * (x_segment - x_nodes[i]) ** 3, color = 'blue')


    plt.scatter(x_nodes, y_nodes, color='red')
    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.xlabel('x')
    plt.ylabel('S(x)')
    plt.grid()
    plt.xlim(min(x_nodes) - 50, max(x_nodes) + 50)
    plt.ylim(min(y_nodes) - 100, max(y_nodes) + 100)


def Plot_parametric_spline(x_nodes, y_nodes, second_derivative_right):

    x_nodes.append(x_nodes[0])
    y_nodes.append(y_nodes[0])

    N = len(x_nodes) - 1
    t, S_1, S_2 = Parametric_spline(x_nodes, y_nodes, second_derivative_right)

    for i in range(N):
        t_segment = np.linspace(t[i], t[i + 1], 100)
        x_segment = S_1[i, 0] + S_1[i, 1] * (t_segment - t[i]) + S_1[i, 2] * (t_segment - t[i]) ** 2 + S_1[i, 3] * (t_segment - t[i]) ** 3
        y_segment = S_2[i, 0] + S_2[i, 1] * (t_segment - t[i]) + S_2[i, 2] * (t_segment - t[i]) ** 2 + S_2[i, 3] * (t_segment - t[i]) ** 3
        plt.plot(x_segment, y_segment, color = 'blue')

    plt.grid()
    plt.scatter(x_nodes, y_nodes, color = 'red')
    plt.xlim(min(x_nodes)-50, max(x_nodes)+50)
    plt.ylim(min(y_nodes)-100, max(y_nodes)+100)


def Plot_first_derivative_spline(x_nodes, y_nodes, second_derivative_right):

    N = len(x_nodes) - 1
    S = Spline(x_nodes, y_nodes, second_derivative_right)

    for i in range(N):
        x_segment = np.linspace(x_nodes[i], x_nodes[i + 1], 100)
        plt.plot(x_segment, S[i, 1] + 2 * S[i, 2] * (x_segment - x_nodes[i]) + 3 * S[i, 3] * (x_segment - x_nodes[i]) ** 2, color='red')

    plt.grid()
    plt.axhline(y = 0, color = 'black')
    plt.axvline(x = 0, color = 'black')
    plt.xlabel('x')
    plt.ylabel('$dS(x)/dx$')


def Plot_second_derivative_spline(x_nodes, y_nodes, second_derivative_right):

    N = len(x_nodes) - 1
    S = Spline(x_nodes, y_nodes, second_derivative_right)

    for i in range(N):
        x_segment = np.linspace(x_nodes[i], x_nodes[i + 1], 100)
        plt.plot(x_segment, 2 * S[i, 2] + 6 * S[i, 3] * (x_segment - x_nodes[i]), color='red')

    plt.axhline(y=0, color='black')
    plt.axvline(x=0, color='black')
    plt.xlabel('x')
    plt.ylabel('$d^2S(x)/dx^2$')
    plt.grid()


x_up  = [34, 50, 79, 114, 152, 190, 215, 236, 259, 293, 330, 374, 420, 464, 494, 534, 572, 615, 632]
y_up = [-elem for elem in [278, 262, 259, 259, 258, 258, 256, 236, 211, 204, 202, 202, 206, 226, 242, 246, 250, 256, 285]]

x = [34, 50, 79, 114, 152, 190, 215, 236, 259, 293, 330, 374, 420, 464, 494, 534, 572, 615, 632, 617, 576, 535, 511, 502, 486, 461, 439, 423, 395, 355, 311, 264, 219, 171, 157, 140, 113, 90, 79, 50, 33]

y = [-elem for elem in [278, 262, 259, 259, 258, 258, 256, 236, 211, 204, 202, 202, 206, 226, 242, 246, 250, 256, 285, 312, 318, 320, 321, 337, 356, 360, 353, 333, 332, 333, 333, 332, 333, 331, 345, 360, 363, 353, 329, 323, 301]]


plt.figure(1)
Plot_spline(x_up, y_up, 0)
plt.figure(2)
Plot_parametric_spline(x, y ,0)
plt.figure(3)
Plot_first_derivative_spline(x_up, y_up, 0)
plt.figure(4)
Plot_second_derivative_spline(x_up, y_up, 0)
plt.figure(5)
Lagrange(x_up, y_up)
plt.figure(6)
plt.scatter(x_up, y_up, color = 'red')
plt.xlim(min(x_up)-50, max(x_up)+50)
plt.ylim(min(y_up)-100, max(y_up)+100)
plt.grid()

plt.show()

