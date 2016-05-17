// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int test_omp_taskwait()
{
  int result1 = 0;   /* Stores number of not finished tasks after the taskwait */
  int result2 = 0;   /* Stores number of wrong array elements at the end */
  int array[NUM_TASKS];
  int i;

  /* fill array */
  for (i = 0; i < NUM_TASKS; i++)
    array[i] = 0;

  #pragma omp parallel
  {
    #pragma omp single
    {
      for (i = 0; i < NUM_TASKS; i++) {
        /* First we have to store the value of the loop index in a new variable
         * which will be private for each task because otherwise it will be overwritten
         * if the execution of the task takes longer than the time which is needed to
         * enter the next step of the loop!
         */
        int myi;
        myi = i;
        #pragma omp task
        {
          my_sleep (SLEEPTIME);
          array[myi] = 1;
        } /* end of omp task */
      } /* end of for */
      #pragma omp taskwait
      /* check if all tasks were finished */
      for (i = 0; i < NUM_TASKS; i++)
        if (array[i] != 1)
          result1++;

      /* generate some more tasks which now shall overwrite
       * the values in the tids array */
      for (i = 0; i < NUM_TASKS; i++) {
        int myi;
        myi = i;
        #pragma omp task
        {
          array[myi] = 2;
        } /* end of omp task */
      } /* end of for */
    } /* end of single */
  } /*end of parallel */

  /* final check, if all array elements contain the right values: */
  for (i = 0; i < NUM_TASKS; i++) {
    if (array[i] != 2)
      result2++;
  }
  return ((result1 == 0) && (result2 == 0));
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_taskwait()) {
      num_failed++;
    }
  }
  return num_failed;
}
