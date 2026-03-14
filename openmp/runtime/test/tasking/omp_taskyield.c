// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int test_omp_taskyield()
{
  int i;
  int count = 0;
  int start_tid[NUM_TASKS];
  int current_tid[NUM_TASKS];

  for (i=0; i< NUM_TASKS; i++) {
    start_tid[i]=0;
    current_tid[i]=0;
  }

  #pragma omp parallel
  {
    #pragma omp single
    {
      for (i = 0; i < NUM_TASKS; i++) {
        int myi = i;
        #pragma omp task untied
        {
          my_sleep(SLEEPTIME);
          start_tid[myi] = omp_get_thread_num();
          #pragma omp taskyield
          if((start_tid[myi] %2) ==0){
            my_sleep(SLEEPTIME);
            current_tid[myi] = omp_get_thread_num();
          } /*end of if*/
        } /* end of omp task */
      } /* end of for */
    } /* end of single */
  } /* end of parallel */
  for (i=0;i<NUM_TASKS; i++) {
    //printf("start_tid[%d]=%d, current_tid[%d]=%d\n",
      //i, start_tid[i], i , current_tid[i]);
    if (current_tid[i] == start_tid[i])
      count++;
  }
  return (count<NUM_TASKS);
}

int main()
{
  int i;
  int num_failed=0;

  if (omp_get_max_threads() < 2)
    omp_set_num_threads(8);

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_taskyield()) {
      num_failed++;
    }
  }
  return num_failed;
}
