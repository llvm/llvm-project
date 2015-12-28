// RUN: %libomp-compile-and-run
#include "omp_testsuite.h"
#include <stdlib.h>
#include <stdio.h>

static int i;
#pragma omp threadprivate(i)

int test_omp_threadprivate_for()
{
  int known_sum;
  int sum;

  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  sum = 0;

  #pragma omp parallel
  {
    int sum0 = 0, i0;
    #pragma omp for
    for (i0 = 1; i0 <= LOOPCOUNT; i0++) {
      i = i0;
      sum0 = sum0 + i;
    }
    #pragma omp critical
    {
      sum = sum + sum0;
    }
  } /* end of parallel */  

  if (known_sum != sum ) {
    fprintf(stderr, " known_sum = %d, sum = %d\n", known_sum, sum);
  }
  return (known_sum == sum);
} /* end of check_threadprivate*/

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_threadprivate_for()) {
      num_failed++;
    }
  }
  return num_failed;
}
