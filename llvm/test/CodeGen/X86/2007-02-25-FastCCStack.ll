; RUN: llc < %s -mtriple=i686-- -mcpu=pentium3

define internal fastcc double @ggc_rlimit_bound(double %limit) {
    ret double %limit
}
