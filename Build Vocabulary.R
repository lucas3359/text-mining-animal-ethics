

######################################## Categorize Keywords ########################################

library(dplyr)
library(data.table)

dataf <- read.csv("J:/Lucas/Animal Research/query/with faculty - 2.csv",header = TRUE)

data_liggins<-dataf%>%filter(FACULTY_CODE == 'LIGGINS')

data_liggins2<-data_liggins%>% distinct(RESEARCH_OUTPUT_ID,.keep_all = TRUE)

Data<- data_liggins2%>%mutate(Text = paste(P_TITLE,ABSTRACT) )

Encoding(Data$Text) <- "UTF-8"
data <- as.character(Data$Text)

### Vocabulary

words <- read.csv("J:/Lucas/Animal Research/specienames2.csv",header = TRUE)
word <- as.character(words$SPECIESNAME)


### Pick up publications containing the keywords

#initialize dataframe
df<-data.frame(matrix(ncol=0,nrow=length(word)))

for (j in 1:length(data)) {
  df[j]<- as.data.frame(ifelse(sapply(1:length(word),function(x)grepl(paste("\\b",word[x],"\\b",sep=""),data[j],ignore.case=TRUE)),1,0))
}


#preparing data for training model
df_backup<-df
rownames(df) <- word
colnames(df) <- Data$P_TITLE
Sum<-colSums (df, na.rm = FALSE, dims = 1)
df<-data.frame(t(rbind(df,Sum)))
setnames(df, "X84", "Sum")
df<-df%>%mutate(animal_related =  ifelse(Sum>0,1,0))
df$RESEARCH_OUTPUT_ID <- Data$RESEARCH_OUTPUT_ID
df<-df %>% dplyr::select(RESEARCH_OUTPUT_ID,animal_related, Sum, everything())
Result_key <- merge(Data,df,by="RESEARCH_OUTPUT_ID", all.x = TRUE)%>%dplyr::select(RESEARCH_OUTPUT_ID, animal_related, P_TITLE, everything())
Result_key2011<-Result_key%>%filter(PUB_YEAR =='2011')
Result<-Result_key%>%filter(PUB_YEAR !='2011')
write.csv(Result,"J:/Lucas/Animal Research/query/Liggins faculty/Liggins_exclude2011.csv")




######################################## Train Model ########################################

### data preparation for text mining model ###

library(quanteda)
library(caret)
library(tm)
library(tidytext)
library(tidyverse)
library(widyr)


Result$animal_related <- as.factor(Result$animal_related)

#construct a corpus
corpus <- corpus(Result, text_field = 'Text')

#for each document, assign a document id in the corpus
docvars(corpus, "id_numeric") <- 1:ndoc(corpus)
head(docvars, 10)


# tokenize text for doing n-grams
token <-corpus%>%
  tokens(
    remove_numbers = TRUE,
    remove_punct = TRUE,
    remove_symbols = TRUE,
    remove_twitter = TRUE,
    remove_url = TRUE,
    remove_hyphens = TRUE,
    include_docvars = TRUE
  )%>%tokens_select(
    c("[\\d-]", "[[:punct:]]", "^.{1,2}$"),
    selection = "remove",
    valuetype = "regex",
    verbose = TRUE
  )%>%tokens_remove(c(stopwords("english"),stop_words$word))

mydfm <- dfm(token,
             tolower = TRUE,
             remove = c(stopwords("english"),stop_words$word))%>%
  dfm_select( min_nchar = 2)%>%
  dfm_trim( max_docfreq = 0.3, docfreq_type = "prop")


df_mydfm<-as.data.frame(t(mydfm))

# 2-grams
token_2grams<-tokens_ngrams(token, n = 2)%>%dfm()%>%dfm_trim( min_termfreq = 5)
x<-as.data.frame(t(token_2grams))

# 3-grmas
token_3grams<-tokens_ngrams(token, n = 3)%>%dfm()%>%dfm_trim( min_termfreq = 5)
x2<-as.data.frame(t(token_3grams))


#enlarge vocabularies with words and some common set-phrase from 2grams & 3grams
dfm<-cbind(mydfm,token_2grams,token_3grams)

dfm_x<-as.data.frame(t(dfm))



### validated faculty`s publication`preparation ###

x <- read.csv("J:/Lucas/Animal Research/query/Liggins faculty/Liggins2011.csv",header = TRUE)
x$Title<-as.character(x$P_TITLE)
x$Abstract<-as.character(x$ABSTRACT)
x$Text<-as.character(x$Text)

x$animal_related<-as.factor(x$animal_related)
Encoding(x$Title) <- "UTF-8"
Encoding(x$Abstract) <- "UTF-8"

corpus_test<-corpus(x, text_field = 'Text')

d1.dfm<-corpus_test%>%tokens(
  remove_numbers = TRUE,
  remove_punct = TRUE,
  remove_symbols = TRUE,
  remove_twitter = TRUE,
  remove_url = TRUE,
  remove_hyphens = TRUE,
  include_docvars = TRUE)%>%
  tokens_select(
    c("[\\d-]", "[[:punct:]]", "^.{1,2}$"),
    selection = "remove",
    valuetype = "regex",
    verbose = TRUE)%>%
  tokens_remove(c(stopwords("english"),stop_words$word))%>%
  tokens_ngrams( n = 1:3)%>%
  dfm()%>%dfm_select( min_nchar = 2)

dfmat_matched <- dfm_match(d1.dfm, features = featnames(dfm))



#### Train model 1 ###

set.seed(100)
nb.classifier <- textmodel_nb(dfm, docvars(corpus, "animal_related"))
predicted<-predict(nb.classifier,newdata = dfmat_matched)

x<-cbind(x,predicted)

actual_class <- docvars(corpus_test, "animal_related")
tab_class <- table(actual_class, predicted)
confusionMatrix(tab_class, mode = "everything")



### Train model 2 ###

library(naivebayes)
set.seed(100)
train_data <- as.matrix(dfm)
test_data <- as.matrix(dfmat_matched)


model<-multinomial_naive_bayes(train_data, docvars(corpus, "animal_related"))


predict<-predict(model, test_data)


x$predict <-predict
x<- x%>%select(RESEARCH_OUTPUT_ID,animal_related,predict,P_TITLE,everything())

confusionMatrix(predict , docvars(corpus_test, "animal_related"))
table(predict(model, test_data),docvars(corpus_test, "Theme"))


write.csv(x,"J:/Lucas/Animal Research/query/Liggins faculty/use liggins to predict 2011.csv")



