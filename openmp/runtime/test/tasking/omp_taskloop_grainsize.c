// RUN: %libomp-compile-and-run
// RUN: %libomp-compile && env KMP_TASKLOOP_MIN_TASKS=1 %libomp-run
// REQUIRES: openmp-4.5

// These compilers don't support the taskloop construct
// UNSUPPORTED: gcc-4, gcc-5, icc-16
// GCC 6 has support for taskloops, but at least 6.3.0 is crashing on this test
// UNSUPPORTED: gcc-6

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

int test_omp_taskloop_grainsize()
{
  int result = 0;
  int i, grainsize, count, tmp_count, num_off;
  int *tmp, *tids, *tidsArray;

  tidsArray = (int *)malloc(sizeof(int) * CFDMAX_SIZE);
  tids = tidsArray;

  for (grainsize = 1; grainsize < 48; ++grainsize) {
    fprintf(stderr, "Grainsize %d\n", grainsize);
    count = tmp_count = num_off = 0;

    for (i = 0; i < CFDMAX_SIZE; ++i) {
      tids[i] = -1;
    }

    #pragma omp parallel shared(tids)
    {
      #pragma omp master
      #pragma omp taskloop grainsize(grainsize)
      for (i = 0; i < CFDMAX_SIZE; i++) {
        tids[i] = omp_get_thread_num();
      }
    }

    for (i = 0; i < CFDMAX_SIZE; ++i) {
      if (tids[i] == -1) {
        fprintf(stderr, "  Iteration %d not touched!\n", i);
        result++;
      }
    }

    for (i = 0; i < CFDMAX_SIZE - 1; ++i) {
      if (tids[i] != tids[i + 1]) {
        count++;
      }
    }

    tmp = (int *)malloc(sizeof(int) * (count + 1));
    tmp[0] = 1;

    for (i = 0; i < CFDMAX_SIZE - 1; ++i) {
      if (tmp_count > count) {
        printf("--------------------\nTestinternal Error: List too "
               "small!!!\n--------------------\n");
        break;
      }
      if (tids[i] != tids[i + 1]) {
        tmp_count++;
        tmp[tmp_count] = 1;
      } else {
        tmp[tmp_count]++;
      }
    }

    // is grainsize statement working?
    int num_tasks = CFDMAX_SIZE / grainsize;
    int multiple1 = CFDMAX_SIZE / num_tasks;
    int multiple2 = CFDMAX_SIZE / num_tasks + 1;
    for (i = 0; i < count; i++) {
      // it is possible for 2 adjacent chunks assigned to a same thread
      if (tmp[i] % multiple1 != 0 && tmp[i] % multiple2 != 0) {
        num_off++;
      }
    }

    if (num_off > 1) {
      fprintf(stderr, "  The number of bad chunks is %d\n", num_off);
      result++;
    } else {
      fprintf(stderr, "  Everything ok\n");
    }

    free(tmp);
  }
  free(tidsArray);
  return (result==0);
}

int main()
{
  int i;
  int num_failed=0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_taskloop_grainsize()) {
      num_failed++;
    }
  }
  return num_failed;
}
