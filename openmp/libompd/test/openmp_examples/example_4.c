// RUN: %gdb-compile-and-run 2>&1 | tee %t.out | FileCheck %s

#include "../ompt_plugin.h"
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

int main() {
  printf("Application: Process %d started.\n", getpid());
  omp_set_num_threads(3);
  omp_set_max_active_levels(10);

#pragma omp parallel // parallel region begins
  {
    printf("Outer region - thread_ID: %d\n", omp_get_thread_num());

#pragma omp parallel num_threads(2) // nested parallel region 1
    {
      printf("Inner region - thread_ID: %d\n", omp_get_thread_num());

#pragma omp parallel num_threads(2) // nested parallel region 2
      {
        int i;
#pragma omp for
        for (i = 0; i < 4; i++)
          printf("Thread %i of %i working on %i\n", omp_get_thread_num(),
                 omp_get_max_threads(), i);
        ompd_tool_test(0);
      }
    }
  }

  return 0;
}

// CHECK-NOT: OMPT-OMPD mismatch
// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
