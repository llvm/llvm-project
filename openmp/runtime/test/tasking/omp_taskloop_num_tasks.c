// RUN: %libomp-compile-and-run
// RUN: %libomp-compile && env KMP_TASKLOOP_MIN_TASKS=1 %libomp-run
// UNSUPPORTED: gcc
// We do not yet have the GOMP interface for taskloop
/*
 * Test for taskloop
 * Method: caculate how many times the iteration space is dispatched
 *     and judge if each dispatch has the requested grainsize
 * It is possible for two adjacent chunks are executed by the same thread
 */
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "omp_testsuite.h"

#define CFDMAX_SIZE 1120

int test_omp_taskloop_num_tasks()
{
  int i;
  int *tids;
  int *tidsArray;
  int count;
  int result = 0;
  int num_tasks;

  for (num_tasks = 1; num_tasks < 120; ++num_tasks) {
    count = 0;
    tidsArray = (int *)malloc(sizeof(int) * CFDMAX_SIZE);
    tids = tidsArray;

    #pragma omp parallel shared(tids)
    {
      int i;
      #pragma omp master
      #pragma omp taskloop num_tasks(num_tasks)
      for (i = 0; i < CFDMAX_SIZE; i++) {
        tids[i] = omp_get_thread_num();
      }
    }

    for (i = 0; i < CFDMAX_SIZE - 1; ++i) {
      if (tids[i] != tids[i + 1]) {
        count++;
      }
    }

    if (count > num_tasks) {
      fprintf(stderr, "counted too many tasks: (wanted %d, got %d)\n",
              num_tasks, count);
      result++;
    }
  }

  return (result==0);
}

int main()
{
  int i;
  int num_failed=0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_taskloop_num_tasks()) {
      num_failed++;
    }
  }
  return num_failed;
}
