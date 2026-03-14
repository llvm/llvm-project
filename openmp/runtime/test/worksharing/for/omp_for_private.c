// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

/* Utility function do spend some time in a loop */
static void do_some_work()
{
  int i;
  double sum = 0;
  for(i = 0; i < 1000; i++){
  sum += sqrt ((double) i);
  }
}

int sum1;
#pragma omp threadprivate(sum1)

int test_omp_for_private()
{
  int sum = 0;
  int sum0;
  int known_sum;

  sum0 = 0;  /* setting (global) sum0 = 0 */

  #pragma omp parallel
  {
    sum1 = 0;  /* setting sum1 in each thread to 0 */
    {  /* begin of orphaned block */
      int i;
      #pragma omp for private(sum0) schedule(static,1)
      for (i = 1; i <= LOOPCOUNT; i++) {
        sum0 = sum1;
        #pragma omp flush
        sum0 = sum0 + i;
        do_some_work ();
        #pragma omp flush
        sum1 = sum0;
      }
    }  /* end of orphaned block */

    #pragma omp critical
    {
      sum = sum + sum1;
    }  /*end of critical*/
  }  /* end of parallel*/
  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  return (known_sum == sum);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_private()) {
      num_failed++;
    }
  }
  return num_failed;
}
