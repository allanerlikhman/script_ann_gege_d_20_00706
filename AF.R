## Factor Analysis##
##choice of directory##
rm(list = ls())
setwd("...")
##Packages##
library(MVN)
library(goft)
library(psych)
library(rela)
library(Rcmdr)
library(MVar.pt)
library(mvShapiroTest)
library(royston)
library(ggplot2)
library(ggcorrplot)
library(psych)
##open the dataset##
data <- read.csv2("morro_da_mina_rmr_teste03_3.csv",header=TRUE,dec = ",")
head(data)
A = data[3:8]
X = as.matrix(A)
## Bartlett sphericity ##
Bartlett.sphericity.test <- function(x)
{
  method <- "Bartlett's test of sphericity"
  data.name <- deparse(substitute(x))
  x <- subset(x, complete.cases(x)) # Omit missing values
  n <- nrow(x)
  p <- ncol(x)
  chisq <- (1-n+(2*p+5)/6)*log(det(cor(x)))
  df <- p*(p-1)/2
  p.value <- pchisq(chisq, df, lower.tail=FALSE)
  names(chisq) <- "X-squared"
  names(df) <- "df"
  return(structure(list(statistic=chisq, parameter=df, p.value=p.value,
                        method=method, data.name=data.name), class="htest"))
}
Bartlett.sphericity.test(A)
## KMO Test ##
R = cor(X)
KMO(R)
## Kaiser criterion ##
k = sum(eigen(R)$values>=1)
k
#AF#
af = FA(A, method = 'PC', nfactor = 3)
## verification of the communality and specificity of each variable ##
h2 = af$mtxcomuna
h2
psi = af$mtxvaresp
psi
## evaluation of the matrix of factor loadings for the interpretation of the factors. 
L = af$mtxcarga
L
## Matrix of factor loadings after rotation in model
rot1.1 = varimax(L)
rot1.1
