// RUN: %libomp-compile-and-run | FileCheck %s

#include <omp.h>
#include <stdio.h>

/* This code tests that state pushed for the num_threads clause does not
   reach the next parallel region. omp_get_max_threads() + 1 can never
   be chosen as team size for the second parallel and could only be the
   result of some left-over state from the first parallel.
 */

int main(int argc, char *argv[]) {
  unsigned N = omp_get_max_threads();
#pragma omp parallel num_threads(N + 1) if (0)
#pragma omp single
  { printf("BBB %2d\n", omp_get_num_threads()); }

#pragma omp parallel
#pragma omp single
  {
    if (omp_get_num_threads() <= N)
      printf("PASS\n");
  }
  return 0;
}

// CHECK: PASS
