// RUN: %libomp-compile-and-run | FileCheck %s

#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  unsigned N = omp_get_max_threads() - 1;
#pragma omp parallel num_threads(N) if (0)
#pragma omp single
  { printf("BBB %2d\n", omp_get_num_threads()); }

#pragma omp parallel
#pragma omp single
  {
    if (omp_get_num_threads() != N)
      printf("PASS\n");
  }
  return 0;
}

// CHECK: PASS
