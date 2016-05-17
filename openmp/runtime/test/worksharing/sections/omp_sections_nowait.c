// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int test_omp_sections_nowait()
{
  int result;
  int count;
  int j;

  result = 0;
  count = 0;

  #pragma omp parallel
  {
    int rank;
    rank = omp_get_thread_num ();
    #pragma omp sections nowait
    {
      #pragma omp section
      {
        fprintf(stderr, "Thread nr %d enters first section"
          " and gets sleeping.\n", rank);
        my_sleep(SLEEPTIME);
        count = 1;
        fprintf(stderr, "Thread nr %d woke up an set"
          " count to 1.\n", rank);
        #pragma omp flush(count)
      }
      #pragma omp section
      {
        fprintf(stderr, "Thread nr %d executed work in the"
          " first section.\n", rank);
      }
    }
    /* Begin of second sections environment */
    #pragma omp sections
    {
      #pragma omp section
      {
        fprintf(stderr, "Thread nr %d executed work in the"
          " second section.\n", rank);
      }
      #pragma omp section
      {
        fprintf(stderr, "Thread nr %d executed work in the"
          " second section and controls the value of count\n", rank);
        if (count == 0)
          result = 1;
        fprintf(stderr, "count was %d\n", count);
      }
    }
  }
  return result;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_sections_nowait()) {
      num_failed++;
    }
  }
  return num_failed;
}
