// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

int test_omp_task_firstprivate()
{
  int i;
  int sum = 1234;
  int known_sum;
  int result = 0; /* counts the wrong sums from tasks */

  known_sum = 1234 + (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;

  #pragma omp parallel
  {
    #pragma omp single
    {
      for (i = 0; i < NUM_TASKS; i++) {
        #pragma omp task firstprivate(sum)
        {
          int j;
          for (j = 0; j <= LOOPCOUNT; j++) {
            #pragma omp flush
            sum += j;
          }

          /* check if calculated sum was right */
          if (sum != known_sum) {
            #pragma omp critical
            { result++; }
          }
        } /* omp task */
      } /* for loop */
    } /* omp single */
  } /* omp parallel */
  return (result == 0);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_task_firstprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
