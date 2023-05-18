// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

int sum0;
#pragma omp threadprivate(sum0)

int test_omp_for_lastprivate()
{
  int sum = 0;
  int known_sum;
  int i0;

  i0 = -1;

  #pragma omp parallel
  {
    sum0 = 0;
    {  /* Begin of orphaned block */
      int i;
      #pragma omp for schedule(static,7) lastprivate(i0)
      for (i = 1; i <= LOOPCOUNT; i++) {
        sum0 = sum0 + i;
        i0 = i;
      }  /* end of for */
    }  /* end of orphaned block */

    #pragma omp critical
    {
      sum = sum + sum0;
    }  /* end of critical */
  }  /* end of parallel */

  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  fprintf(stderr, "known_sum = %d , sum = %d\n",known_sum,sum);
  fprintf(stderr, "LOOPCOUNT = %d , i0 = %d\n",LOOPCOUNT,i0);
  return ((known_sum == sum) && (i0 == LOOPCOUNT));
}

int main()
{
  int i;
  int num_failed=0;

  for (i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_lastprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
