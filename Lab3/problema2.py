import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


x = sp.symbols('x')
expr = x**5 - 8*x**3 + 10*x + 6

# Diferenciar 
expr_diff = sp.diff(expr, x)
expr_second_diff = sp.diff(expr_diff, x)  

# Crear funciones desde las expresiones
f = sp.lambdify(x, expr, 'numpy')
f_first = sp.lambdify(x, expr_diff, 'numpy') # primera derivada
f_second = sp.lambdify(x, expr_second_diff, 'numpy') # segunda derivada


#---- Metodo Newton Raphson ----#
def newton_raphson(x0, factor_converg, tolerance=0.0001, max_iter=100):
    i = 0
    x_i = x0
    while abs(f_first(x_i)) > tolerance and i < max_iter:

        if abs(f_second(x_i)) <= 0.0001: # No podemos permitir division entre 0
            break

        # Computar siguiente aproximacion:
        x_next = x_i - factor_converg * (f_first(x_i) / f_second(x_i))
        
        # Siguiente iteracion
        x_i = x_next
        i += 1

    
    return x_i





# ---- Test con diferentes puntos ----#
puntos_intervalo = [-3, -2, 1, 0, 1, 2, 3]
extremos = []

for x0 in puntos_intervalo:
    punto = newton_raphson(x0, factor_converg=0.6) 
    if f_second(punto) > 0:
        tipo = "min"
    else:
        tipo = "max"

    extremos.append((punto, f(punto), tipo))


# Maximo y minimo global
max_global = max(extremos, key=lambda x: x[1])  
min_global = min(extremos, key=lambda x: x[1])  



# ---- GRAFICO ----
plt.figure(figsize=(8, 6))
x_vals = np.linspace(-3, 3, 1000)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals)

# Extremos locales
for x_ext, y_ext, tipo in extremos:
    plt.plot(x_ext, y_ext, 'go', markersize=8, label=f'{tipo.capitalize()} local en x={x_ext:.4f}')


plt.plot(max_global[0], max_global[1], 'ro', markersize=10, label=f'Max global en x={max_global[0]:.4f}')
plt.plot(min_global[0], min_global[1], 'ro', markersize=10, label=f'Min global en x={min_global[0]:.4f}')


plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.savefig('Lab3/images/Problema2.png')
plt.show()



