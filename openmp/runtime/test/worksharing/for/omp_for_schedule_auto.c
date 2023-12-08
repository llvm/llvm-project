// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp_testsuite.h"

int sum1;
#pragma omp threadprivate(sum1)

int test_omp_for_auto()
{
  int j;
  int sum;
  int sum0;
  int known_sum;
  int threadsnum;

  sum = 0;
  sum0 = 12345;

  // array which keeps track of which threads participated in the for loop
  // e.g., given 4 threads, [ 0 | 1 | 1 | 0 ] implies
  //       threads 0 and 3 did not, threads 1 and 2 did
  int max_threads = omp_get_max_threads();
  int* active_threads = (int*)malloc(sizeof(int)*max_threads);
  for(j = 0; j < max_threads; j++)
    active_threads[j] = 0;

  #pragma omp parallel
  {
    int i;
    sum1 = 0;
    #pragma omp for firstprivate(sum0) schedule(auto)
    for (i = 1; i <= LOOPCOUNT; i++) {
      active_threads[omp_get_thread_num()] = 1;
      sum0 = sum0 + i;
      sum1 = sum0;
    }

    #pragma omp critical
    {
      sum = sum + sum1;
    }
  }

  // count the threads that participated (sum is stored in threadsnum)
  threadsnum=0;
  for(j = 0; j < max_threads; j++) {
    if(active_threads[j])
      threadsnum++;
  }
  free(active_threads);

  known_sum = 12345 * threadsnum + (LOOPCOUNT * (LOOPCOUNT + 1)) / 2;
  return (known_sum == sum);
}

int main()
{
  int i;
  int num_failed=0;

  for(i = 0; i < REPETITIONS; i++) {
    if(!test_omp_for_auto()) {
      num_failed++;
    }
  }
  return num_failed;
}
