// RUN: %llvmgcc %s -o /dev/null -S
// Note:
//  We fail this on Sparc because the C library seems to be missing complex.h
//  and the corresponding C99 complex support.
//
//  We could modify the test to use only GCC extensions, but I don't know if
//  that would change the nature of the test.
//
// XFAIL: sparc

#ifdef __CYGWIN__
  #include <mingw/complex.h>
#else
  #include <complex.h>
#endif

int foo(complex float c) {
    return creal(c);
}
