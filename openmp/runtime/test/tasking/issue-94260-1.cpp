// RUN: %libomp-cxx-compile-and-run

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// The number of times to run each test
#define NTIMES 2

// Every thread creates a single "increment" task
void test_tasks() {
  for (int i = 0; i < 100; ++i)
#pragma omp task
  {
    int tid = omp_get_thread_num();
  }
}

// Testing single level of parallelism with increment tasks
void test_base(int nthreads) {
#ifdef VERBOSE
#pragma omp master
  printf("    test_base(%d)\n", nthreads);
#endif
#pragma omp parallel num_threads(nthreads)
  { test_tasks(); }
}

// Testing nested parallel with increment tasks
// first = nthreads of outer parallel
// second = nthreads of nested parallel
void test_nest(int first, int second) {
#ifdef VERBOSE
#pragma omp master
  printf("   test_nest(%d, %d)\n", first, second);
#endif
#pragma omp parallel num_threads(first)
  {
    for (int i = 0; i < 100; ++i)
#pragma omp task
    {
      int tid = omp_get_thread_num();
    }
    test_base(second);
  }
}

template <typename... Args>
void run_ntimes(int n, void (*func)(Args...), Args... args) {
  for (int i = 0; i < n; ++i) {
    func(args...);
  }
}

int main() {
  omp_set_max_active_levels(5);

  for (int i = 0; i < 100; ++i) {
    run_ntimes(NTIMES, test_nest, 4, 3);
    run_ntimes(NTIMES, test_nest, 2, 1);
  }

  printf("PASS\n");
  return EXIT_SUCCESS;
}
