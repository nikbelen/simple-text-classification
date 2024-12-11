library(dplyr)         # Работа с данными
library(textstem)      # Лемматизация
library(stringr)       # Строки
library(tm)            # Векторизация
library(caret)         # Создание моделей машинного обучения
library(e1071)         # Метод опорных векторов 
library(randomForest)  # RandomForest
library(ggplot2)
library(reshape2)

# Подгружаем таблицы CSV
fact_clean <- read.csv("files/politifact_clean.csv")
fact_clean_bi <- read.csv("files/politifact_clean_binarized.csv")

data <- fact_clean
data <- select(data, -link)

#
# Предобработка текста и классов
#

# нормирование текста
clean_text_function <- function(text) {
  text %>%
    tolower() %>%                           # Приведение к нижнему регистру
    str_replace_all("[[:punct:]]", "") %>%  # Удаление пунктуации
    str_squish() %>%                        # Удаление лишних пробелов
    lemmatize_strings()                     # Лемматизация
}
data <- data %>%
  mutate(clean_text = clean_text_function(statement))

# кодирование значений
data$class <- as.factor(data$veracity)
data$class_bi <- as.factor(ifelse(fact_clean_bi$veracity == 1, "True", "False"))

# векторизация
corpus <- Corpus(VectorSource(data$clean_text))
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, 0.99)
dtm_matrix <- as.data.frame(as.matrix(dtm))
dtm_matrix$class <- data$class
dtm_matrix$class_bi <- data$class_bi


#
# Подготовка к обучению
#

# многоклассовая
set.seed(123)

train_index <- createDataPartition(dtm_matrix$class, p = 0.7, list = FALSE)
train_data <- dtm_matrix[train_index, ]
test_data <- dtm_matrix[-train_index, ]

# бинарная
set.seed(123)

train_index <- createDataPartition(dtm_matrix$class_bi, p = 0.7, list = FALSE)
train_data_bi <- dtm_matrix[train_index, ]
test_data_bi <- dtm_matrix[-train_index, ]


#
# RandomForest
#

# многоклассовая
rf_model <- randomForest(x = select(train_data, -class, -class_bi),
                         y = train_data$class,
                         ntree = 100)
predictions_rf <- predict(rf_model,
                          newdata = select(test_data, -class, -class_bi))
conf_matrix_rf_multi <- confusionMatrix(predictions_rf,
                                        test_data$class)
print(conf_matrix_rf_multi)     # Accuracy : 0.2922

# бинарная
rf_model <- randomForest(x = select(train_data_bi, -class, -class_bi),
                         y = train_data_bi$class_bi,
                         ntree = 100)
predictions_rf <- predict(rf_model,
                          newdata = select(test_data_bi, -class, -class_bi))
conf_matrix_rf_bi <- confusionMatrix(predictions_rf,
                                     test_data_bi$class_bi)
print(conf_matrix_rf_bi)        # Accuracy : 0.6396


#
# Метод опорных векторов
#

# многоклассовая
svm_model <- svm(x = select(train_data, -class, -class_bi),
                 y = train_data$class,
                 kernel = "linear")
predictions_svm <- predict(svm_model, newdata = select(test_data, -class, -class_bi))
conf_matrix_svm_multi <- confusionMatrix(predictions_svm, test_data$class)
print(conf_matrix_svm_multi)   # Accuracy : 0.2821

svm_model <- svm(x = select(train_data, -class, -class_bi),
                 y = train_data$class,
                 kernel = "polynomial",
                 degree = 2)
predictions_svm <- predict(svm_model, newdata = select(test_data, -class, -class_bi))
conf_matrix_svm_multi_pol <- confusionMatrix(predictions_svm, test_data$class)
print(conf_matrix_svm_multi_pol)   # Accuracy : (3) 0.2648, (2) 0.2829

# бинарная
svm_model <- svm(x = select(train_data_bi, -class, -class_bi),
                 y = train_data_bi$class_bi,
                 kernel = "linear")
predictions_svm <- predict(svm_model, newdata = select(test_data_bi, -class, -class_bi))
conf_matrix_svm_bi <- confusionMatrix(predictions_svm, test_data_bi$class_bi)
print(conf_matrix_svm_bi)      # Accuracy : 0.62


#
# Графики
#

draw_function <- function(conf_matrix, text) {
  conf_table_melted <- melt(as.data.frame(conf_matrix$table))
  
  ggplot(data = conf_table_melted, aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = value), color = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(label = value), vjust = 1) +
    theme_minimal() +
    labs(title = text, x = "Actual", y = "Predicted") +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
} 

draw_function(conf_matrix_rf_multi, "RandomForest, multi")
draw_function(conf_matrix_rf_bi, "RandomForest, bi")
draw_function(conf_matrix_svm_multi, "SVM, multi-lin")
draw_function(conf_matrix_svm_multi_pol, "SVM, multi-pol")
draw_function(conf_matrix_svm_bi, "SVM, bi-lin")
