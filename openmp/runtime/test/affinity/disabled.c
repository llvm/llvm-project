// RUN: %libomp-compile
// RUN: env KMP_AFFINITY=disabled %libomp-run
// RUN: env KMP_AFFINITY=disabled,reset %libomp-run
// REQUIRES: affinity
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
  int nthreads, correct_value;;
  int a = 0;
  #pragma omp parallel reduction(+: a)
  {
    a += omp_get_thread_num();
    #pragma omp single
    nthreads = omp_get_num_threads();
  }
  correct_value = nthreads * (nthreads - 1) / 2;
  if (a != correct_value) {
    printf("Incorrect value: %d should be %d\n", a, correct_value);
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

