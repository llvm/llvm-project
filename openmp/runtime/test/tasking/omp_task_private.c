// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

/* Utility function do spend some time in a loop */
int test_omp_task_private()
{
  int i;
  int known_sum;
  int sum = 0;
  int result = 0; /* counts the wrong sums from tasks */

  known_sum = (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;

  #pragma omp parallel
  {
    #pragma omp single
    {
      for (i = 0; i < NUM_TASKS; i++) {
        #pragma omp task private(sum) shared(result, known_sum)
        {
          int j;
          //if sum is private, initialize to 0
          sum = 0;
          for (j = 0; j <= LOOPCOUNT; j++) {
            #pragma omp flush
            sum += j;
          }
          /* check if calculated sum was right */
          if (sum != known_sum) {
            #pragma omp critical
            result++;
          }
        } /* end of omp task */
      } /* end of for */
    } /* end of single */
  } /* end of parallel*/
  return (result == 0);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_task_private()) {
      num_failed++;
    }
  }
  return num_failed;
}
