import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


x = sp.symbols('x')
expr = (3*x)**3 - (10*x)**2 - 56*x + 50

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
        # Computar siguiente aproximacion:
        x_next = x_i - factor_converg * (f_first(x_i) / f_second(x_i))
        
        # Siguiente iteracion:
        x_i = x_next
        i += 1

    
    return x_i





# ---- Test con diferentes puntos ----#
puntos_intervalo = [-6, -4, -2, 0, 2, 4, 6]
extremos = []

for x0 in puntos_intervalo:
    # Encontrar extremo
    punto = newton_raphson(x0, factor_converg=0.6)
    
    # Clasificar como min o max local
    if f_second(punto) > 0:
        minimo_bool = True
    else:
        minimo_bool = False
    
    # Guardar resultado
    extremos.append((punto, minimo_bool))




# ---- GRAFICO ----
plt.figure(figsize=(8, 6))
x_vals = np.linspace(-6, 6, 1000)
y_vals = f(x_vals)
plt.plot(x_vals, y_vals)


extremos_unicos = []

# Filtrar extremos duplicados
for extremo, es_minimo in extremos:
    extremos_unicos.append((extremo, es_minimo))

# Graficar cada extremo 
for extremo, es_minimo in extremos_unicos:
    if es_minimo:
        plt.plot(extremo, f(extremo), 'go', markersize=8, label=f'Mínimo en x = {extremo:.10f}')
    else:
        plt.plot(extremo, f(extremo), 'ro', markersize=8, label=f'Máximo en x = {extremo:.10f}')

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.savefig('Lab3/images/Problema1.png')
plt.show()