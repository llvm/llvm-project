// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int test_omp_task_if()
{
  int condition_false;
  int count;
  int result;

  count=0;
  condition_false = (count == 1);
  #pragma omp parallel
  {
    #pragma omp single
    {
      #pragma omp task if (condition_false) shared(count, result)
      {
        my_sleep (SLEEPTIME);
        #pragma omp critical
        result = (0 == count);
      } /* end of omp task */
      #pragma omp critical
      count = 1;
    } /* end of single */
  } /*end of parallel */
  return result;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_task_if()) {
      num_failed++;
    }
  }
  return num_failed;
}
