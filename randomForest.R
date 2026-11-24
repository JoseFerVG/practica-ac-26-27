
# =====================================================================
# Práctica 1 AC — Busqueda de hiperparámetros Random Forest
# =====================================================================

library(caret)

credit <- readRDS("credit_limpio.rds") #Leer los datos de train limpios

train <- credit$train
test <- credit$test


# Ya que caret ofrece muy pocos hiperparametros para Random Forest.
# En concreto, para method = rf, solo está mtry, que es el número de variables 
# predictoras seleccionadas aleatoriamente en cada división de nodo.
# Como solo tenemos 15 variables predictoras, podemos probar todos los valores
# sin que suponga un coste excesivo
# También está ntree, el número de arboles del bosque, pero caret no lo 
# incluye en su búsqueda automática.


# Dado que tenemos un conjunto equilibrado (325 no, 261 si) la medida accuracy
# sera más relevante qe Kappa.

# Primero, voy a optimizar mtry

grid_rf <- expand.grid(mtry = 1:15) # Creamos el grid

# Establecemos la semilla
set.seed(42)

# Creamos el control, cross-validation con 10 particiones
control <- trainControl(
  method = "cv",
  number = 10
)

lista_modelo_rf <- vector("list", 5)
#Entrenar
for(i in 1:5){
  lista_modelo_rf[[i]] <- train(
    Granted ~ .,
    data = train,
    method = "rf",
    tuneGrid = grid_rf,
    ntree = 500,
    trControl = control
  )
}


# 1. Juntar los resultados de los 5 modelos
resultados <- do.call(rbind, lapply(lista_modelo_rf, function(m) m$results))

# 2. Media de accuracy por cada mtry
media_acc <- aggregate(Accuracy ~ mtry, data = resultados, FUN = mean)
media_acc

ggplot(media_acc, aes(x = factor(mtry), y = Accuracy)) + geom_col(fill = "blue") +
  labs(x = "mtry", y = "Media Accuracy",
       title = "Accuracy medio por valor de mtry") +
  coord_cartesian(ylim = c(0.775, 0.875)) 
# Como se puede ver en la gráfica, el valor de accuracy para cada ntree 
# está muy ajustado, pero parece ser que 4  es el que mejor resultado da.

# Fijamos como valor de mtry = 4

grid_rf <- expand.grid(mtry = 4)

# El ntree = 500 ya está probado da este valor de media.
media_acc[4,]
# 0.8702281


lista_ntree <- c(50, 100, 250, 750, 1000)

# Establecemos la semilla
set.seed(42)
for(i in 1:5){
  lista_modelo_rf[[i]] <- train(
    Granted ~ .,
    data = train,
    method = "rf",
    tuneGrid = grid_rf,
    ntree = lista_ntree[i],
    trControl = control
  )
}

resultados <- do.call(rbind, lapply(lista_modelo_rf, function(m) m$results))
resultados[1:2]

# ntree = 250 obtiene el mejor accuracy, asi que buscamos alrededor de ese valor

lista_ntree <- c(200, 225, 250, 275, 300)

lista_resultados <- vector("list", 20)
# Establecemos la semilla
set.seed(42)
for(j in 1:20){
  for(i in 1:5){
    lista_modelo_rf[[i]] <- train(
      Granted ~ .,
      data = train,
      method = "rf",
      tuneGrid = grid_rf,
      ntree = lista_ntree[i],
      trControl = control
    )
  }
  lista_resultados[[j]] <- do.call(rbind, lapply(lista_modelo_rf, function(m) m$results))
}
# Organizarlo todo en filas
resultados <- do.call(rbind, lista_resultados)
# Poner los valores de ntree
resultados$ntree <- rep(lista_ntree, times = 20)       

# Hacer la media de Accuracy por tree
media_acc <- aggregate(Accuracy ~ ntree, data = resultados, FUN = mean)
media_acc

ggplot(media_acc, aes(x = factor(ntree), y = Accuracy)) + geom_col(fill = "green") +
  labs(x = "ntree", y = "Media Accuracy",
       title = "Accuracy medio por valor de ntree") +
  coord_cartesian(ylim = c(0.86, 0.875)) 

# El que obtiene más Accuracy medio tras 20 ejecuciones es ntree = 200, como se 
# puede apreciar en la gráfica

ntree_final = 200

# Quitamos el metodo, ya hemos escogido los hiperparámetros
control <- trainControl(
  method = "none"
)

# Establecemos la semilla, para que sea reproducible
set.seed(42)
modelo_final <- train(
  Granted ~ .,
  data = train,
  method = "rf",
  tuneGrid = grid_rf,
  ntree = ntree_final,
  trControl = control
)

pred <- predict(modelo_final, newdata = test)

confusionMatrix(pred, test$Granted)