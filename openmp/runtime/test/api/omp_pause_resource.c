// RUN: %libomp-compile-and-run

// Linking fails for icc 18/19
// UNSUPPORTED: icc-18, icc-19

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include "omp_testsuite.h"

#define NUM_THREADS 3

void doOmpWorkWithCritical(int *a_lockCtr, int *b_lockCtr) {
#pragma omp parallel num_threads(NUM_THREADS)
  {
#pragma omp critical(a_lock)
    { *a_lockCtr = *a_lockCtr + 1; }
#pragma omp critical(b_lock)
    { *b_lockCtr = *b_lockCtr + 1; }
  }
}

void test_omp_critical_after_omp_hard_pause_resource_all() {
  int a_lockCtr = 0, b_lockCtr = 0;

  // use omp to do some work
  doOmpWorkWithCritical(&a_lockCtr, &b_lockCtr);
  assert(a_lockCtr == NUM_THREADS && b_lockCtr == NUM_THREADS);
  a_lockCtr = b_lockCtr = 0; // reset the counters

  // omp hard pause should succeed
  int rc = omp_pause_resource_all(omp_pause_hard);
  assert(rc == 0);

  // we should not segfault inside the critical sections of doOmpWork()
  doOmpWorkWithCritical(&a_lockCtr, &b_lockCtr);
  assert(a_lockCtr == NUM_THREADS && b_lockCtr == NUM_THREADS);
}

void test_omp_get_thread_num_after_omp_hard_pause_resource_all() {
  // omp_get_thread_num() should work, even if omp is not yet initialized
  int n = omp_get_thread_num();
  // called from serial region, omp_get_thread_num() should return 0
  assert(n == 0);

// use omp to do some work, guarantees omp initialization
#pragma omp parallel num_threads(NUM_THREADS)
  {}

  // omp hard pause should succeed
  int rc = omp_pause_resource_all(omp_pause_hard);
  assert(rc == 0);

  // omp_get_thread_num() should work again with no segfault
  n = omp_get_thread_num();
  // called from serial region, omp_get_thread_num() should return 0
  assert(n == 0);
}

void test_omp_parallel_num_threads_after_omp_hard_pause_resource_all() {
// use omp to do some work
#pragma omp parallel num_threads(NUM_THREADS)
  {}

  // omp hard pause should succeed
  int rc = omp_pause_resource_all(omp_pause_hard);
  assert(rc == 0);

// this should not trigger any omp asserts
#pragma omp parallel num_threads(NUM_THREADS)
  {}
}

void test_KMP_INIT_AT_FORK_with_fork_after_omp_hard_pause_resource_all() {
  // explicitly set the KMP_INIT_AT_FORK environment variable to 1
  setenv("KMP_INIT_AT_FORK", "1", 1);

// use omp to do some work
#pragma omp parallel for num_threads(NUM_THREADS)
  for (int i = 0; i < NUM_THREADS; ++i) {
  }

  // omp hard pause should succeed
  int rc = omp_pause_resource_all(omp_pause_hard);
  assert(rc == 0);

// use omp to do some work
#pragma omp parallel for num_threads(NUM_THREADS)
  for (int i = 0; i < NUM_THREADS; ++i) {
  }

  // we'll fork .. this shouldn't deadlock
  int p = fork();

  if (!p) {
    exit(0); // child simply does nothing and exits
  }

  waitpid(p, NULL, 0);

  unsetenv("KMP_INIT_AT_FORK");
}

void test_fork_child_exiting_after_omp_hard_pause_resource_all() {
// use omp to do some work
#pragma omp parallel num_threads(NUM_THREADS)
  {}

  // omp hard pause should succeed
  int rc = omp_pause_resource_all(omp_pause_hard);
  assert(rc == 0);

  int p = fork();

  if (!p) {
    // child should be able to exit properly without assert failures
    exit(0);
  }

  waitpid(p, NULL, 0);
}

int test_omp_pause_resource() {
  int fails, nthreads, my_dev;

  fails = 0;
  nthreads = 0;
  my_dev = omp_get_initial_device();

#pragma omp parallel
#pragma omp single
  nthreads = omp_get_num_threads();

  if (omp_pause_resource(omp_pause_soft, my_dev))
    fails++;

#pragma omp parallel shared(nthreads)
#pragma omp single
  nthreads = omp_get_num_threads();

  if (nthreads == 0)
    fails++;
  if (omp_pause_resource(omp_pause_hard, my_dev))
    fails++;
  nthreads = 0;

#pragma omp parallel shared(nthreads)
#pragma omp single
  nthreads = omp_get_num_threads();

  if (nthreads == 0)
    fails++;
  if (omp_pause_resource_all(omp_pause_soft))
    fails++;
  nthreads = 0;

#pragma omp parallel shared(nthreads)
#pragma omp single
  nthreads = omp_get_num_threads();

  if (nthreads == 0)
    fails++;
  return fails == 0;
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_pause_resource()) {
      num_failed++;
    }
    test_omp_critical_after_omp_hard_pause_resource_all();
    test_omp_get_thread_num_after_omp_hard_pause_resource_all();
    test_omp_parallel_num_threads_after_omp_hard_pause_resource_all();
    test_KMP_INIT_AT_FORK_with_fork_after_omp_hard_pause_resource_all();
    test_fork_child_exiting_after_omp_hard_pause_resource_all();
  }
  return num_failed;
}
