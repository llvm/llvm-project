; RUN: llc < %s -march=x86 -mcpu=pentium3

define internal fastcc double @ggc_rlimit_bound(double %limit) {
    ret double %limit
}
