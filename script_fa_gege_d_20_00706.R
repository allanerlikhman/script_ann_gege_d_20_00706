## Script to Factor analysis model ##
## To run the script, it is necessary that the database has the weight system adopted by the present research (according to Table 1)##
## Clearing the R memory ##
rm(list = ls())
## choice of work directory ##
## Here the user chooses the folder where his database is and where the R history and the outputs will be saved ##
setwd("C:/Users/....")
## Packages used ##
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
## Choose your database in csv format ##
## If there is any difficulty in reading the database, please consult the documentation at "help(read.csv2)" ##
data <- read.csv2("        ")
## Transforming the database into an array ##
X = as.matrix(data)
## Applying the Bartlett sphericity test ##
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
## Applying the KMO test ##
R = cor(X)
KMO(R)
## Determination of the number of factors: Kaiser's criterion ##
k = sum(eigen(R)$values>=1)
eigen(R)$values
eigen(R)
k
## Applying Factor analysis ##
## Extraction of factors using the principal component method ##
af = FA(A, method = 'PC', nfactor = 3)
## Communalities result
h2 = af$mtxcomuna
h2
## Specificity result
psi = af$mtxvaresp
psi
## Factorial load matrix
L= af$mtxcarga
L
## Rotation of the factor load matrix using varimax
rot = varimax(L)
rot
