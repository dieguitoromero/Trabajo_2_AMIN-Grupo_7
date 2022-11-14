# Trabajo_2_AMIN-Grupo_7
Implementación del sistema de colonia de hormigas de Marco Dorigo.

## Creado por:
- Jeremy Aguirre Dumenes
- Diego Romero Carrillo

## Requerimientos:
- Python 3.6.5+
- Scipy, pandas, numpy: `python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose`

## Ejecucion:
Para los comandos definidos en clase: `python AntColony.py`

Para valores personalizados: `python AntColonySystem.py tamaño_colonia feromona heuristica probabilidad max_iteraciones archivo_entrada`

Para valores personalizados y semilla: `python AntColonySystem.py tamaño_colonia feromona heuristica probabilidad max_iteraciones archivo_entrada semilla`

## Ejemplo con parametros:
`python AntColonySystem.py 100 0.1 2.5 0.9 500 berlin52.tsp 123`
