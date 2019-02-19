// RUN: %libomp-compile && env KMP_AFFINITY=compact %libomp-run
// REQUIRES: openmp-4.0

#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "omp_testsuite.h"

int test_nested_affinity_bug() {
  int a = 0;
  omp_set_nested(1);
  #pragma omp parallel num_threads(2) shared(a)
  {
    #pragma omp parallel num_threads(2) shared(a) proc_bind(close)
    {
      #pragma omp atomic
      a++;
    }
  }
  return 1;
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_nested_affinity_bug()) {
      num_failed++;
    }
  }
  return num_failed;
}
