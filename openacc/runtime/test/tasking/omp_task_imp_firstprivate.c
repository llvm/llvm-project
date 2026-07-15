// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <math.h>
#include "omp_testsuite.h"

/* Utility function do spend some time in a loop */
int test_omp_task_imp_firstprivate()
{
  int i=5;
  int k = 0;
  int result = 0;
  int task_result = 1;
  #pragma omp parallel firstprivate(i)
  {
    #pragma omp single
    {
      for (k = 0; k < NUM_TASKS; k++) {
        #pragma omp task shared(result , task_result)
        {
          int j;
          //check if i is private
          if(i != 5)
            task_result = 0;
          for(j = 0; j < NUM_TASKS; j++)
            i++;
          //this should be firstprivate implicitly
        }
      }
      #pragma omp taskwait
      result = (task_result && i==5);
    }
  }
  return result;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_task_imp_firstprivate()) {
      num_failed++;
    }
  }
  return num_failed;
}
