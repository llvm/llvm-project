// RUN: %llvmgcc -S %s -o - | grep readnone
// PR2520
#include <math.h>
double f(double *x, double *y) { return fabs(*x + *y); }
