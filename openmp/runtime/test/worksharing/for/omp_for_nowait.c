// RUN: %libomp-compile-and-run
#include <stdio.h>
#include "omp_testsuite.h"
#include "omp_my_sleep.h"

int test_omp_for_nowait()
{
  int result;
  int count;
  int j;
  int myarray[LOOPCOUNT];

  result = 0;
  count = 0;

  #pragma omp parallel
  {
    int rank;
    int i;

    rank = omp_get_thread_num();

    #pragma omp for nowait
    for (i = 0; i < LOOPCOUNT; i++) {
      if (i == 0) {
        my_sleep(SLEEPTIME);
        count = 1;
        #pragma omp flush(count)
      }
    }

    #pragma omp for
    for (i = 0; i < LOOPCOUNT; i++) {
      #pragma omp flush(count)
      if (count == 0)
        result = 1;
    }
  }
  return result;
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_nowait()) {
      num_failed++;
    }
  }
  return num_failed;
}
