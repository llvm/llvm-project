// RUN: %llvmgcc %s -o /dev/null -S
// Note:
//  We fail this on SparcV9 because the C library seems to be missing complex.h
//  and the corresponding C99 complex support.
//
//  We could modify the test to use only GCC extensions, but I don't know if
//  that would change the nature of the test.
//
// XFAIL: sparc

#include <complex.h>

int foo(complex float c) {
    return creal(c);
}
