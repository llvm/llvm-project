// RUN: %libomp-compile
// RUN: env KMP_TASKING=0 %libomp-run
// RUN: env KMP_TASKING=1 %libomp-run
// RUN: env KMP_TASKING=2 %libomp-run
//
// Test to make sure the KMP_TASKING=1 option doesn't crash
// Can use KMP_TASKING=0 (immediate exec) or 2 (defer to task queue
// and steal during regular barrier) but cannot use
// KMP_TASKING=1 (explicit tasking barrier before regular barrier)
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main() {
  int i;
#pragma omp parallel
  {
#pragma omp single
    {
      for (i = 0; i < 10; i++) {
#pragma omp task
        {
          printf("Task %d executed by thread %d\n", i, omp_get_thread_num());
        }
      }
    }
  }
  return EXIT_SUCCESS;
}
