// RUN: %llvmgcc %s -S -o -

#include <math.h>
#define I 1.0iF

double __complex test(double X) { return ~-(X*I); }

_Bool EQ(double __complex A, double __complex B) { return A == B; }
_Bool NE(double __complex A, double __complex B) { return A != B; }
