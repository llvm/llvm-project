// RUN: %gdb-compile-and-run 2>&1 | tee %t.out | FileCheck %s

#include "../ompt_plugin.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  int n = 5;
  if (argc > 1)
    n = atoi(argv[1]);
  int i = 0;
  int a[1000];
#pragma omp parallel for
  for (i = 0; i < 100; ++i) {
#pragma omp task
    {
      a[i] = 42;
      ompd_tool_test(0);
    }
  }
  return 0;
}

// CHECK-NOT: OMPT-OMPD mismatch
// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
