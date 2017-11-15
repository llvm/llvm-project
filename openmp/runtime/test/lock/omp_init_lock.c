// RUN: %libomp-compile-and-run
#include "omp_testsuite.h"
#include <stdio.h>

// This should be slightly less than KMP_I_LOCK_CHUNK, which is 1024
#define LOCKS_PER_ITER 1000
#define ITERATIONS (REPETITIONS + 1)

// This tests concurrently using locks on one thread while initializing new
// ones on another thread.  This exercises the global lock pool.
int test_omp_init_lock() {
  int i;
  omp_lock_t lcks[ITERATIONS * LOCKS_PER_ITER];
#pragma omp parallel for schedule(static) num_threads(NUM_TASKS)
  for (i = 0; i < ITERATIONS; i++) {
    int j;
    omp_lock_t *my_lcks = &lcks[i * LOCKS_PER_ITER];
    for (j = 0; j < LOCKS_PER_ITER; j++) {
      omp_init_lock(&my_lcks[j]);
    }
    for (j = 0; j < LOCKS_PER_ITER * 100; j++) {
      omp_set_lock(&my_lcks[j % LOCKS_PER_ITER]);
      omp_unset_lock(&my_lcks[j % LOCKS_PER_ITER]);
    }
  }
  // Wait until all repititions are done.  The test is exercising growth of
  // the global lock pool, which does not shrink when no locks are allocated.
  {
    int j;
    for (j = 0; j < ITERATIONS * LOCKS_PER_ITER; j++) {
      omp_destroy_lock(&lcks[j]);
    }
  }

  return 0;
}

int main() {
  // No use repeating this test, since it's exercising a private global pool
  // which is not reset between test iterations.
  return test_omp_init_lock();
}
