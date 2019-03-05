getwd()
setwd("C:/Projetos/CompeticaoDSA_201901")

## Analise exploratoria dos dados

# Carregando os dados
df = read.csv('dataset_treino.csv', stringsAsFactors = F)

# Fazendo analise dos dados
head(df)
str(df)


cols <- c("num_gestacoes", "glicose", "pressao_sanguinea",
          "grossura_pele", "insulina", "bmi",
          "indice_historico", "idade", "correlacao")

# Verificando a correlacao entre as variaveis preditoras
df$correlacao <- df$classe - predict(lm(classe ~ ., data = df), newdata = df)

metodos <- c("pearson", "spearman")

cors <- lapply(metodos, function(method)(cor(df[, cols], method = method)))

head(cors)

require(lattice)
plot.cors <- function(x, labs) {
  diag(x) <- 0.0
  plot(levelplot(x, main = paste("Plot de correlacao usando Metodo", labs),
                 scales = list(x = list(rot = 90), cex = 1.0)))
}

Map(plot.cors, cors, metodos)

dim(df)
any(is.na(df))


cols2 <- c("num_gestacoes", "glicose", "pressao_sanguinea",
          "grossura_pele", "insulina", "bmi",
          "indice_historico", "idade")

# Normalizacao
normalizar <- function(df, variaveis){
  for (var in variaveis){
    df[[var]] <- scale(df[[var]], center = T, scale = T)
  }
  return(df)
}


df <- normalizar(df, cols2)

str(df)



# Dividindo os dados em treino e teste - 60:40
index <- sample(1:nrow(df), size = 0.6 * nrow(df))
dadosTreino <- df[index, ]
dadosTeste <- df[-index, ]

str(dadosTreino)
str(dadosTeste)

# Feature Selection
install.packages("Matrix")
install.packages("caret")
install.packages("randomForest")
library(caret)
library(randomForest)

# Funcao para selecao de variaveis
run.feature.selection <- function(num.iters = 10, feature.vars, class.var){
  set.seed(10)
  variable.sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv",
                        verbose = FALSE, returnResamp = "all",
                        number = num.iters)
  results.rfe <- rfe(x = feature.vars, y = class.var,
                     sizes = variable.sizes,
                     rfeControl = control)
  return(results.rfe)
}

# Executando a funcao
rfe.results <- run.feature.selection(feature.vars = dadosTreino[,-1], class.var = dadosTreino[,1])

# Visualizando os resultados
rfe.results


# Criando e avaliando o modelo
library(caret)
install.packages("ROCR")
library(ROCR)

# Separate feature and class variables
test.feature.vars <- dadosTeste[2:9]
test.class.var <- dadosTeste[,10]

# Construindo um modelo de regrssao logistica
formula.int <- "classe ~ ."
formula.int <- as.formula(formula.int)
lr.model <- glm(formula = formula.int, data = dadosTreino, family = "binomial")

# Visualizando o modelo
summary(lr.model)


# Testando o modelo nos dados de teste
lr.predictions <- predict(lr.model, dadosTeste, type = "response")
lr.predictions <- round(lr.predictions)

# Avaliando o modelo
confusionMatrix(table(data = lr.predictions, reference = test.class.var), positive = '1')


# Feature selection
formula <- "classe ~ num_gestacoes + bmi + glicose"
formula <- as.formula(formula)
lr.model.new <- glm(formula = formula, data = dadosTreino, family = "gaussian")

# Visualizando o modelo
summary(lr.model.new)

# Testando o modelo nos dados de teste
lr.predictions.new <- predict(lr.model.new, dadosTeste, type = "response")
lr.predictions.new <- round(lr.predictions.new)

# Avaliando o modelo
confusionMatrix(table(data = lr.predictions.new, reference = test.class.var), positive = '1')

?glm


############## Preparando para envio

dadosEnvio = read.csv("dataset_teste.csv", stringsAsFactors = F)


dadosEnvio <- normalizar(dadosEnvio, cols2)


dadosEnvio$classe <- predict(lr.model.new, dadosEnvio, type = "response")
str(dadosEnvio)
dadosEnvio$classe <- round(dadosEnvio$classe)
str(dadosEnvio)
dadosEnvioAlterado <- dadosEnvio[,-c(2:9)]
str(dadosEnvioAlterado)
dadosEnvioAlterado


write.csv(dadosEnvioAlterado, file = "data/Submission_v1.csv", sep = ",", row.names = FALSE)
