sc <- read.table(file('stdin'))
scratio <- sc$V2 / sc$V3
scratio <- scratio[scratio > 0]

# Take the log of the ratio. Our null hypothesis is a normal distribution
# around zero.
tt <- t.test(log(scratio))
tt

# This gives us the geo-mean as we are taking the exponent of the linear mean
# of logarithms.
1 - 1/exp(tt$estimate)

# Likewise for the confidence interval.
1 - 1/exp(tt$conf.int)
