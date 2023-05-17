// RUN: %gdb-compile-and-run 2>&1 | tee %t.out | FileCheck %s

#include "../ompt_plugin.h"
#include <omp.h>
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>

void createPthreads() {
  int numThreads = 2;
  pthread_t threads[numThreads];
  int i;
  for (i = 0; i < numThreads; ++i)
    pthread_create(&threads[i], NULL, ompd_tool_break, NULL);

  for (i = 0; i < numThreads; ++i)
    pthread_join(threads[i], NULL);
}

int main() {
  omp_set_num_threads(4);
  printf("Application: Process %d started.\n", getpid());
  createPthreads(); // thread_data is set to 0x0 if called

// Parallel region 1
#pragma omp parallel
  { ompd_tool_test(0); }

  return 0;
}

// CHECK-NOT: OMPT-OMPD mismatch
// CHECK-NOT: Python Exception
// CHECK-NOT: The program is not being run.
