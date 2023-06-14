// RUN: %libomp-compile-and-run | FileCheck %s

#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
#pragma omp parallel num_threads(3) if (false)
#pragma omp single
  { printf("BBB %2d\n", omp_get_num_threads()); }

#pragma omp parallel
#pragma omp single
  { printf("CCC %2d\n", omp_get_num_threads()); }
  return 0;
}

// CHECK-NOT: CCC  3
