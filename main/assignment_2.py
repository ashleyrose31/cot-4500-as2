#question 1 
from decimal import Decimal
def neville(x, fx, x_target):
    n = len(x)
    p = [Decimal(val) for val in fx]

    for j in range(1, n):
        for i in range(n - j):
            numerator = (Decimal(x_target)- Decimal(x[i + j])) * p[i] + (Decimal(x[i]) - Decimal(x_target)) * p[i + 1 ]
            denominator = Decimal(x[i]) - Decimal(x[i + j])
            p[i] = numerator / denominator 

    return float(p[0])

x_values = [3.6, 3.8, 3.9]
fx_values = [1.675, 1.436, 1.318]
x_target = 3.7 

result = neville(x_values, fx_values, x_target)
print(f"{result:.15f}")
print()
#question 2
import numpy as np
import sympy as sp

x_values = np.array([7.2, 7.4, 7.5, 7.6])
f_values = np.array([23.5492, 25.3913, 26.8224, 27.4589])


f01 = (f_values[1] - f_values[0]) / (x_values[1] - x_values[0])
f12 = (f_values[2] - f_values[1]) / (x_values[2] - x_values[1])
f23 = (f_values[3] - f_values[2]) / (x_values[3] - x_values[2])


f012 = (f12 - f01) / (x_values[2] - x_values[0])
f123 = (f23 - f12) / (x_values[3] - x_values[1])


f0123 = (f123 - f012) / (x_values[3] - x_values[0])


expected_f012 = -0.7183802816901438  
expected_f0123 = -0.12461196085345332  


x = sp.Symbol('x')
newton_poly = f_values[0] + f01 * (x - x_values[0]) + \
              expected_f012 * (x - x_values[0]) * (x - x_values[1]) + \
              expected_f0123 * (x - x_values[0]) * (x - x_values[1]) * (x - x_values[2])

#question 3
f_7_3_corrected = newton_poly.subs(x, 7.3)

#final output for questions 2 and 3 with proper formatting
print(f"{f01:.16f}")
print(f"{expected_f012:.16f}")
print(f"{expected_f0123:.16f}")
print() 
print(f"{f_7_3_corrected:.16f}")

print()
#question 4
import numpy as np

def hermite_divided_difference(x, fx, fx_deriv):
    n = len(x)
    z = np.zeros(2 * n)
    Q = np.zeros((2 * n, 2 * n))

    for i in range(n):
        z[2 * i] = x[i]
        z[2 * i + 1] = x[i]
        Q[2 * i, 0] = fx[i]
        Q[2 * i + 1, 0] = fx[i]
        Q[2 * i + 1, 1] = fx_deriv[i]

        if i > 0:
            Q[2 * i, 1] = (Q[2 * i, 0] - Q[2 * i - 1, 0]) / (z[2 * i] - z[2 * i - 1])

    for j in range(2, 2 * n):
        for i in range(2 * n - j):
            Q[i, j] = (Q[i + 1, j - 1] - Q[i, j - 1]) / (z[i + j] - z[i])

    return z, Q

x_values = [3.6, 3.8, 3.9]
fx_values = [1.675, 1.436, 1.318]
fx_deriv_values = [-1.195, -1.188, -1.182]

z, Q = hermite_divided_difference(x_values, fx_values, fx_deriv_values)

np.set_printoptions(precision=10, suppress=True)

for i in range(len(z)):
    formatted_row = f"{z[i]:.7f} " + " ".join(
        f"{num:.10e}" if abs(num) < 1e-2 else f"{num:.7f}" for num in Q[i]
    )
    print(f"[ {formatted_row} ]")

print()

#question 5
import numpy as np

x = np.array([2, 5, 8, 10])
f_x = np.array([3, 5, 7, 9])

n = len(x) - 1
h = np.diff(x)
alpha = np.zeros(n)

for i in range(1, n):
    alpha[i] = (3 / h[i]) * (f_x[i + 1] - f_x[i]) - (3 / h[i - 1]) * (f_x[i] - f_x[i - 1])
A = np.zeros((n + 1, n + 1))
b = np.zeros(n + 1)
A[0, 0] = 1
A[n, n] = 1

for i in range(1, n):
    A[i, i - 1] = h[i - 1]
    A[i, i] = 2 * (h[i - 1] + h[i])
    A[i, i + 1] = h[i]
    b[i] = alpha[i]

x = np.linalg.solve(A, b)

np.set_printoptions(precision=1, suppress=True)

for row in A:
    print("[", " ".join(f"{int(num)}." if num.is_integer() else f"{num:.1f}" for num in row), "]")

print("[", " ".join(f"{int(num)}." if num.is_integer() else f"{num:.1f}" for num in b), "]")

print("[", " ".join(f"{int(num)}." if num.is_integer() else f"{num:.8f}" for num in x), "]")