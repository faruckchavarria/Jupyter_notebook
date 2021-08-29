# 4. Probando Numpy y Pandas
#### a. ¿Campo de aplicación de las librerías Pandas y Numpy?
NumPy y Pandas son las librerías de python ideales para análisis de datos y computación numérica. Seguramente también nos enfrentaremos a problemas que no requieren el uso de aprendizaje automático sino sólo el análisis de datos. Por supuesto, también podemos usar estas librerías en estos casos.
#### b. Utilizando la hoja de trucos de Numpy, pruebe cada uno de los códigos y 
muestre los resultados de cada sección:
##### Creando arrays



```python
import numpy as np
a = np.array([1,2,3])
b = np.array([(1.5,2,3), (4,5,6)], dtype = float)
c = np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]], dtype = float)
```

### Creando arrays de ceros 


```python
np.zeros((3,4)) 

```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])



### Creando arrays de uno


```python
np.ones((2,3,4),dtype=np.int16)
```




    array([[[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]],
    
           [[1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]], dtype=int16)



### Creando arrays igualado


```python
d = np.arange(10,25,5)
```


```python
 np.linspace(0,2,9)
```




    array([0.  , 0.25, 0.5 , 0.75, 1.  , 1.25, 1.5 , 1.75, 2.  ])



### Creando arrays constantes, arays con valores random, y de matrices


```python
e = np.full((2,2),7) 
f = np.eye(2)
```


```python
np.random.random((2,2))
```




    array([[0.23652508, 0.510818  ],
           [0.73496005, 0.10276855]])




```python
 np.empty((3,2))
```




    array([[0., 0.],
           [0., 0.],
           [0., 0.]])



### Guardado y cargado de disco


```python
np.save('my_array', a)
np.savez('array.npz', a, b)
np.load('my_array.npy')
```




    array([1, 2, 3])



### Guardado y cargado de archivos de texto


```python
np.loadtxt("myfile.txt")
np.genfromtxt("my_file.csv", delimiter=',')
np.savetxt("myarray.txt", a, delimiter=" ")
```


    ---------------------------------------------------------------------------

    OSError                                   Traceback (most recent call last)

    <ipython-input-30-1b01e021a1d7> in <module>
    ----> 1 np.loadtxt("myfile.txt")
    

    ~\anaconda3\lib\site-packages\numpy\lib\npyio.py in loadtxt(fname, dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding, max_rows, like)
       1063             fname = os_fspath(fname)
       1064         if _is_string_like(fname):
    -> 1065             fh = np.lib._datasource.open(fname, 'rt', encoding=encoding)
       1066             fencoding = getattr(fh, 'encoding', 'latin1')
       1067             fh = iter(fh)
    

    ~\anaconda3\lib\site-packages\numpy\lib\_datasource.py in open(path, mode, destpath, encoding, newline)
        192 
        193     ds = DataSource(destpath)
    --> 194     return ds.open(path, mode, encoding=encoding, newline=newline)
        195 
        196 
    

    ~\anaconda3\lib\site-packages\numpy\lib\_datasource.py in open(self, path, mode, encoding, newline)
        529                                       encoding=encoding, newline=newline)
        530         else:
    --> 531             raise IOError("%s not found." % path)
        532 
        533 
    

    OSError: myfile.txt not found.


# Tipos de datos


```python
np.int64 ##prueba de entero de 64 bits
```




    numpy.int64




```python
np.float32 ##prueba de coma flotante de 32
```




    numpy.float32




```python
np.complex128 ##este nos permite usar valores de tipo coma flotante de mas de 128 bits
```




    numpy.complex128




```python
np.bool_ ## valores boleanos que nos permite mostrar valores de verdarero y falso
```




    numpy.bool_




```python
object ## de tipo objeto
```




    object




```python
np.string_ ## de tipo string nos permite hacer cadenas
```




    numpy.bytes_




```python
np.unicode_ ##usando codigo de un mismo tipo
```




    numpy.str_



### Inspeccion de arrays


```python
a.shape 
len(a) 
b.ndim 
e.size
b.dtype
b.dtype.name
b.astype(int) 
```




    array([[1, 2, 3],
           [4, 5, 6]])



# Pidiendo ayuda


```python
np.info(np.ndarray.dtype)
```

    Data-type of the array's elements.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    d : numpy dtype object
    
    See Also
    --------
    numpy.dtype
    
    Examples
    --------
    >>> x
    array([[0, 1],
           [2, 3]])
    >>> x.dtype
    dtype('int32')
    >>> type(x.dtype)
    <type 'numpy.dtype'>
    

# arrays matematicos, operaciones aritmeticas


```python
 g = a - b ##sustraccion
 ([[-0.5, 0. , 0. ],
 [-3. , -3. , -3. ]])
```




    [[-0.5, 0.0, 0.0], [-3.0, -3.0, -3.0]]




```python
np.subtract(a,b) 
```




    array([[-0.5,  0. ,  0. ],
           [-3. , -3. , -3. ]])




```python
b + a ##Adicion de arrays
([[ 2.5, 4. , 6. ],
[ 5. , 7. , 9. ]])
np.add(b,a) 
```




    array([[2.5, 4. , 6. ],
           [5. , 7. , 9. ]])




```python
a / b ##Division
([[ 0.66666667, 1. , 1. ],
[ 0.25 , 0.4 , 0.5 ]])
np.divide(a,b)
```




    array([[0.66666667, 1.        , 1.        ],
           [0.25      , 0.4       , 0.5       ]])




```python
a * b ##Multiplication
([[ 1.5, 4. , 9. ],
[ 4. , 10. , 18. ]])
np.multiply(a,b)
```




    array([[ 1.5,  4. ,  9. ],
           [ 4. , 10. , 18. ]])




```python
np.exp(b) ##Exponenciacion
```




    array([[  4.48168907,   7.3890561 ,  20.08553692],
           [ 54.59815003, 148.4131591 , 403.42879349]])




```python
np.sqrt(b) ## raiz cuadrada
```




    array([[1.22474487, 1.41421356, 1.73205081],
           [2.        , 2.23606798, 2.44948974]])




```python
 np.sin(a) ##imprimir senos de una matriz
```




    array([0.84147098, 0.90929743, 0.14112001])




```python
 np.cos(b) ##coseno de elementos
```




    array([[ 0.0707372 , -0.41614684, -0.9899925 ],
           [-0.65364362,  0.28366219,  0.96017029]])




```python
 np.log(a) ##logaritmo natural por elementos
```




    array([0.        , 0.69314718, 1.09861229])




```python
 e.dot(f) ##Dot product
([[ 7., 7.],
[ 7., 7.]])
```




    array([[7., 7.],
           [7., 7.]])



# Comparaciones


```python
a == b ##Comparación de elementos
```




    array([[False,  True,  True],
           [False, False, False]])




```python
a < 2 ##
```




    array([ True, False, False])




```python
 np.array_equal(a, b) ## arrays de comparaciond e elementos
```




    False



# Agregar funciones


```python
 a.sum() #sumamos los arreglos
```




    6




```python
a.min() #Matris de valor minimo
```




    1




```python
b.max(axis=0) #valor máximo de una fila de matriz
```




    array([4., 5., 6.])




```python
b.cumsum(axis=1) #suma acumulativa de los elementos
```




    array([[ 1.5,  3.5,  6.5],
           [ 4. ,  9. , 15. ]])




```python
a.mean() #media
```




    2.0




```python
np.median(b)
```




    3.5




```python
np.corrcoef(a)
```




    1.0




```python
np.std(b) #desviacion standard
```




    1.5920810978785667



# Copiar matrices


```python
h = a.view() #crear una vista de la matriz con los mismos datos
```


```python
np.copy(a) #creamos la copia del array
```




    array([1, 2, 3])




```python
h = a.copy() # creando copia profunda de la matriz
```

# Ordenación de matrices



```python
 a.sort() 
```


```python
 c.sort(axis=0) 
```

# Subconjunto, rebanado, indexación


```python
 a[2] #elija el elemento en el segundo índice Esto es un subconjuto
```




    3




```python
 b[1,2] #elija el elemento en la fila 1 columna 2(equivalente a b [1] [2])
```




    6.0




```python
#Esto es un rebanado
a[0:2] #elegir elementos en el índice 0 y 1
```




    array([1, 2])




```python
 b[0:2,1] #elegir elementos en las filas 0 y 1 en la columna 1
```




    array([2., 5.])




```python
 b[:1]  #Seleccionar todos los elementos de la fila 0
```




    array([[1.5, 2. , 3. ]])




```python
 c[1,...] #Lo mismo que [1,:,:]
```




    array([[3., 2., 3.],
           [4., 5., 6.]])




```python
 a[ : :-1] #Matriz invertida a
```




    array([3, 2, 1])



# Indexación booleana


```python
 a[a<2]  #Seleccione elementos de menos de 2
```




    array([1])



# Indexación elegante


```python
 b[[1, 0, 1, 0],[0, 1, 2, 0]] 
```




    array([4. , 2. , 6. , 1.5])




```python
 b[[1, 0, 1, 0]][:,[0,1,2,0]] 
```




    array([[4. , 5. , 6. , 4. ],
           [1.5, 2. , 3. , 1.5],
           [4. , 5. , 6. , 4. ],
           [1.5, 2. , 3. , 1.5]])



# Manipulación de matrices


```python
#Transposición de matriz
i = np.transpose(b) #Permutar las dimensiones de la matriz
i.T
```




    array([[1.5, 2. , 3. ],
           [4. , 5. , 6. ]])




```python
#Cambiar la forma de la matriz
b.ravel() #Aplanar la matriz
```




    array([1.5, 2. , 3. , 4. , 5. , 6. ])




```python
g.reshape(3,-2) #Dale nueva forma, pero no cambies los datos
```




    array([[-0.5,  0. ],
           [ 0. , -3. ],
           [-3. , -3. ]])




```python
#Agregar / eliminar elementos
h.resize((2,6)) #Devuelve una nueva matriz con forma (2,6)
```


```python
 np.append(h,g) #Agregar elementos a una matriz
```




    array([ 1. ,  2. ,  3. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
            0. , -0.5,  0. ,  0. , -3. , -3. , -3. ])




```python
 np.insert(a, 1, 5) #Insertar elementos en una matriz
```




    array([1, 5, 2, 3])




```python
np.delete(a,[1]) #Eliminar elementos de una matriz
```




    array([1, 3])



# Combinando matrices


```python
np.concatenate((a,d),axis=0) #Concatenar matrices
```




    array([ 1,  2,  3, 10, 15, 20])




```python
np.vstack((a,b)) #Apilar matrices verticalmente (por filas)
```




    array([[1. , 2. , 3. ],
           [1.5, 2. , 3. ],
           [4. , 5. , 6. ]])




```python
np.r_[e,f]  #Stack arrays vertically (row-wise)
```




    array([[7., 7.],
           [7., 7.],
           [1., 0.],
           [0., 1.]])




```python
np.hstack((e,f)) # Apilar matrices horizontalmente (en columnas)
```




    array([[7., 7., 1., 0.],
           [7., 7., 0., 1.]])




```python
np.column_stack((a,d)) #Cree matrices apiladas en columnas
```




    array([[ 1, 10],
           [ 2, 15],
           [ 3, 20]])




```python
 np.c_[a,d] #Create stacked column-wise arrays
```




    array([[ 1, 10],
           [ 2, 15],
           [ 3, 20]])



# División de matrices


```python
 np.hsplit(a,3)  #Divida la matriz horizontalmente en el tercer índice
```




    [array([1]), array([2]), array([3])]




```python
np.vsplit(c,2)  #Divida la matriz verticalmente en el segundo índice
```




    [array([[[1.5, 2. , 1. ],
             [4. , 5. , 6. ]]]),
     array([[[3., 2., 3.],
             [4., 5., 6.]]])]




```python

```
