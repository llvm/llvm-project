// RUN: %gdb-compile-and-run 2>&1 | tee %t.out | FileCheck %s

#include "../ompt_plugin.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int fib(int n) {
  int i, j;
  if (n < 2) {
    ompd_tool_test(0);
    return n;
  } else {
#pragma omp task shared(i)
    i = fib(n - 1);
#pragma omp task shared(j)
    j = fib(n - 2);
#pragma omp taskwait
    return i + j;
  }
}

int main(int argc, char **argv) {
  int n = 5;
  if (argc > 1)
    n = atoi(argv[1]);
#pragma omp parallel
  {
#pragma omp single
    printf("fib(%i) = %i\n", n, fib(n));
  }
  return 0;
}

// CHECK-NOT: OMPT-OMPD mismatch
// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
