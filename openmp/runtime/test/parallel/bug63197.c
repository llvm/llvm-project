// RUN: %libomp-compile-and-run | FileCheck %s

#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
#pragma omp parallel num_threads(3) if (0)
#pragma omp single
  { printf("BBB %2d\n", omp_get_num_threads()); }

#pragma omp parallel
#pragma omp single
  {
    if (omp_get_num_threads() != 3)
      printf("PASS\n");
  }
  return 0;
}

// CHECK: PASS
