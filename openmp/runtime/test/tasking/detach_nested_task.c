// RUN: %libomp-compile-and-run

// Checked gcc 10.1 still does not support detach clause on task construct.
// UNSUPPORTED: gcc-4, gcc-5, gcc-6, gcc-7, gcc-8, gcc-9, gcc-10
// gcc 11 introduced detach clause, but gomp interface in libomp has no support
// XFAIL: gcc-11, gcc-12
// clang supports detach clause since version 11.
// UNSUPPORTED: clang-10, clang-9, clang-8, clang-7
// icc compiler does not support detach clause.
// UNSUPPORTED: icc

// The outer detachable task creates multiple child tasks with dependencies
// when the last inner task incremented ret, the task calls omp_fulfill_event
// to release the outer task.

#include <omp.h>
#include <stdio.h>
#include "omp_my_sleep.h"

int *buf;

int foo(int n)
{
  int ret = 0;
  for (int i = 0; i < n; ++i) {
    omp_event_handle_t event;
    #pragma omp task detach(event) firstprivate(i,n) shared(ret)
    {
      for (int j = 0; j < n; ++j) {
        #pragma omp task firstprivate(event,i,j,n) shared(ret) default(none) depend(out:ret)
        {
          //printf("Task %i, %i: %i\n", i, j, omp_get_thread_num());
          my_sleep(.01);
          #pragma omp atomic
            ret++;
#if _OPENMP
          if (j == n-1) {
            //printf("Task %i, %i: omp_fulfill_event()\n", i, j);
            omp_fulfill_event(event);
          }
#endif
        }
      }
    }
  }
  // the taskwait only guarantees the outer tasks to complete.
  #pragma omp taskwait

  return ret;
}


int main()
{
  int ret;
#pragma omp parallel num_threads(4)
#pragma omp master
  {
    ret = foo(8);
  }
  printf("%i\n", ret);
  return !(ret == 64);
}
