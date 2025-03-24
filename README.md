# carteraDeClientesEnMicrofi /..
instalada la librería `seaborn` en Python 3
```makdown
pip3 install seaborn
```
Instala `scikit-learn` con el siguiente comando:
```makdown
pip3 install scikit-learn
```
Instala `imbalanced-learn` con el siguiente comando:
```makdown
pip install imbalanced-learn
```
Si pandas no tiene instalado `openpyxl` (necesario para exportar a Excel), instálalo con:
```makdown
pip install openpyxl
```

# comandos para mergear desde consola
Paso 1 Clone el repositorio o actualice su repositorio local con los `últimos cambios.`

```makdown
git pull origen main
```
Paso 2 `Cambie a la rama base` de la solicitud de extracción.
```makdown
git checkout main
```
Paso 3 Fusiona la rama de la cabeza con la rama base.
```makdown
git merge dev
```
Paso 4 Empuje los cambios.
```makdown
- git push -u origin main
```
# como restaurar un git add
```makdown
git restore
```