// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include "omp_testsuite.h"

//static int sum1 = 789;

int test_omp_parallel_firstprivate()
{
  int sum, num_threads,sum1;
  int known_sum;

  sum = 0;
  sum1=7;
  num_threads = 0;

  #pragma omp parallel firstprivate(sum1)
  {
    /*printf("sum1=%d\n",sum1);*/
    int i;
    #pragma omp for
    for (i = 1; i < 1000; i++) {
      sum1 = sum1 + i;
    } /*end of for*/
    #pragma omp critical
    {
      sum = sum + sum1;
      num_threads++;
    } /*end of critical*/
  } /* end of parallel*/
  known_sum = (999 * 1000) / 2 + 7 * num_threads;
  return (known_sum == sum);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_parallel_firstprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
