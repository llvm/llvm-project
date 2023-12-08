// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

int sum1;
#pragma omp threadprivate(sum1)

int test_omp_for_firstprivate()
{
  int sum;
  int sum0;
  int known_sum;
  int threadsnum;

  sum = 0;
  sum0 = 12345;
  sum1 = 0;

  #pragma omp parallel
  {
    #pragma omp single
    {
      threadsnum=omp_get_num_threads();
    }
    /* sum0 = 0; */

    int i;
    #pragma omp for firstprivate(sum0)
    for (i = 1; i <= LOOPCOUNT; i++) {
      sum0 = sum0 + i;
      sum1 = sum0;
    }  /* end of for */

    #pragma omp critical
    {
      sum = sum + sum1;
    }  /* end of critical */
  }  /* end of parallel */
  known_sum = 12345* threadsnum+ (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  return (known_sum == sum);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_firstprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
