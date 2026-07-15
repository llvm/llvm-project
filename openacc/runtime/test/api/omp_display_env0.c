// RUN: %libomp-compile-and-run 2>&1 | FileCheck %s
// RUN: %libomp-cxx-compile-c && %libomp-run 2>&1 | FileCheck %s
#include <stdio.h>
#include <omp.h>
int main()
{
  omp_display_env(0);
  printf("passed\n");
  return 0;
}

// CHECK: OPENMP DISPLAY ENVIRONMENT BEGIN
// CHECK: _OPENMP
// CHECK: OPENMP DISPLAY ENVIRONMENT END
