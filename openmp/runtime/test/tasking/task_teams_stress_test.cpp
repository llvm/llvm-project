// RUN: %libomp-cxx-compile
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=0 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=1 KMP_HOT_TEAMS_MODE=0 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=1 KMP_HOT_TEAMS_MODE=1 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=2 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=3 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=4 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=5 %libomp-run
//
// RUN: %libomp-cxx-compile -DUSE_HIDDEN_HELPERS=1
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=0 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=1 KMP_HOT_TEAMS_MODE=0 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=1 KMP_HOT_TEAMS_MODE=1 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=2 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=3 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=4 %libomp-run
// RUN: env KMP_HOT_TEAMS_MAX_LEVEL=5 %libomp-run

// This test stresses the task team mechanism by running a simple
// increment task over and over with varying number of threads and nesting.
// The test covers nested serial teams and mixing serial teams with
// normal active teams.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// The number of times to run each test
#define NTIMES 5

// Regular single increment task
void task_inc_a(int *a) {
#pragma omp task
  {
#pragma omp atomic
    (*a)++;
  }
}

// Splitting increment task that binary splits the incrementing task
void task_inc_split_a(int *a, int low, int high) {
#pragma omp task firstprivate(low, high)
  {
    if (low == high) {
#pragma omp atomic
      (*a)++;
    } else if (low < high) {
      int mid = (high - low) / 2 + low;
      task_inc_split_a(a, low, mid);
      task_inc_split_a(a, mid + 1, high);
    }
  }
}

#ifdef USE_HIDDEN_HELPERS
// Hidden helper tasks force serial regions to create task teams
void task_inc_a_hidden_helper(int *a) {
#pragma omp target map(tofrom : a[0]) nowait
  {
#pragma omp atomic
    (*a)++;
  }
}
#else
// Detached tasks force serial regions to create task teams
void task_inc_a_detached(int *a, omp_event_handle_t handle) {
#pragma omp task detach(handle)
  {
#pragma omp atomic
    (*a)++;
    omp_fulfill_event(handle);
  }
}
#endif

void check_a(int *a, int expected) {
  if (*a != expected) {
    fprintf(stderr,
            "FAIL: a = %d instead of expected = %d. Compile with "
            "-DVERBOSE for more verbose output.\n",
            *a, expected);
    exit(EXIT_FAILURE);
  }
}

// Every thread creates a single "increment" task
void test_tasks(omp_event_handle_t *handles, int expected, int *a) {
  int tid = omp_get_thread_num();

  task_inc_a(a);

#pragma omp barrier
  check_a(a, expected);
#pragma omp barrier
  check_a(a, expected);
#pragma omp barrier

#ifdef USE_HIDDEN_HELPERS
  task_inc_a_hidden_helper(a);
#else
  task_inc_a_detached(a, handles[tid]);
#endif

#pragma omp barrier
  check_a(a, 2 * expected);
#pragma omp barrier
  task_inc_a(a);
#pragma omp barrier
  check_a(a, 3 * expected);
}

// Testing single level of parallelism with increment tasks
void test_base(int nthreads) {
#ifdef VERBOSE
#pragma omp master
  printf("    test_base(%d)\n", nthreads);
#endif
  int a = 0;
  omp_event_handle_t *handles;
  handles = (omp_event_handle_t *)malloc(sizeof(omp_event_handle_t) * nthreads);
#pragma omp parallel num_threads(nthreads) shared(a)
  { test_tasks(handles, nthreads, &a); }
  free(handles);
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
  { test_base(second); }
}

// Testing 2-level nested parallels with increment tasks
// first = nthreads of outer parallel
// second = nthreads of nested parallel
// third = nthreads of second nested parallel
void test_nest2(int first, int second, int third) {
#ifdef VERBOSE
#pragma omp master
  printf("  test_nest2(%d, %d, %d)\n", first, second, third);
#endif
#pragma omp parallel num_threads(first)
  { test_nest(second, third); }
}

// Testing 3-level nested parallels with increment tasks
// first = nthreads of outer parallel
// second = nthreads of nested parallel
// third = nthreads of second nested parallel
// fourth = nthreads of third nested parallel
void test_nest3(int first, int second, int third, int fourth) {
#ifdef VERBOSE
#pragma omp master
  printf(" test_nest3(%d, %d, %d, %d)\n", first, second, third, fourth);
#endif
#pragma omp parallel num_threads(first)
  { test_nest2(second, third, fourth); }
}

// Testing 4-level nested parallels with increment tasks
// first = nthreads of outer parallel
// second = nthreads of nested parallel
// third = nthreads of second nested parallel
// fourth = nthreads of third nested parallel
// fifth = nthreads of fourth nested parallel
void test_nest4(int first, int second, int third, int fourth, int fifth) {
#ifdef VERBOSE
#pragma omp master
  printf("test_nest4(%d, %d, %d, %d, %d)\n", first, second, third, fourth,
         fifth);
#endif
#pragma omp parallel num_threads(first)
  { test_nest3(second, third, fourth, fifth); }
}

// Single thread starts a binary splitting "increment" task
// Detached tasks are still single "increment" task
void test_tasks_split(omp_event_handle_t *handles, int expected, int *a) {
  int tid = omp_get_thread_num();

#pragma omp single
  task_inc_split_a(a, 1, expected); // task team A

#pragma omp barrier
  check_a(a, expected);
#pragma omp barrier
  check_a(a, expected);
#pragma omp barrier

#ifdef USE_HIDDEN_HELPERS
  task_inc_a_hidden_helper(a);
#else
  task_inc_a_detached(a, handles[tid]);
#endif

#pragma omp barrier
  check_a(a, 2 * expected);
#pragma omp barrier
#pragma omp single
  task_inc_split_a(a, 1, expected); // task team B
#pragma omp barrier
  check_a(a, 3 * expected);
}

// Testing single level of parallelism with splitting incrementing tasks
void test_base_split(int nthreads) {
#ifdef VERBOSE
#pragma omp master
  printf("  test_base_split(%d)\n", nthreads);
#endif
  int a = 0;
  omp_event_handle_t *handles;
  handles = (omp_event_handle_t *)malloc(sizeof(omp_event_handle_t) * nthreads);
#pragma omp parallel num_threads(nthreads) shared(a)
  { test_tasks_split(handles, nthreads, &a); }
  free(handles);
}

// Testing nested parallels with splitting tasks
// first = nthreads of outer parallel
// second = nthreads of nested parallel
void test_nest_split(int first, int second) {
#ifdef VERBOSE
#pragma omp master
  printf(" test_nest_split(%d, %d)\n", first, second);
#endif
#pragma omp parallel num_threads(first)
  { test_base_split(second); }
}

// Testing doubly nested parallels with splitting tasks
// first = nthreads of outer parallel
// second = nthreads of nested parallel
// third = nthreads of second nested parallel
void test_nest2_split(int first, int second, int third) {
#ifdef VERBOSE
#pragma omp master
  printf("test_nest2_split(%d, %d, %d)\n", first, second, third);
#endif
#pragma omp parallel num_threads(first)
  { test_nest_split(second, third); }
}

template <typename... Args>
void run_ntimes(int n, void (*func)(Args...), Args... args) {
  for (int i = 0; i < n; ++i) {
    func(args...);
  }
}

int main() {
  omp_set_max_active_levels(5);

  run_ntimes(NTIMES, test_base, 4);
  run_ntimes(NTIMES, test_base, 1);
  run_ntimes(NTIMES, test_base, 8);
  run_ntimes(NTIMES, test_base, 2);
  run_ntimes(NTIMES, test_base, 6);
  run_ntimes(NTIMES, test_nest, 1, 1);
  run_ntimes(NTIMES, test_nest, 1, 5);
  run_ntimes(NTIMES, test_nest, 2, 6);
  run_ntimes(NTIMES, test_nest, 1, 1);
  run_ntimes(NTIMES, test_nest, 4, 3);
  run_ntimes(NTIMES, test_nest, 3, 2);
  run_ntimes(NTIMES, test_nest, 1, 1);
  run_ntimes(NTIMES, test_nest2, 1, 1, 2);
  run_ntimes(NTIMES, test_nest2, 1, 2, 1);
  run_ntimes(NTIMES, test_nest2, 2, 2, 1);
  run_ntimes(NTIMES, test_nest2, 2, 1, 1);
  run_ntimes(NTIMES, test_nest2, 4, 2, 1);
  run_ntimes(NTIMES, test_nest2, 4, 2, 2);
  run_ntimes(NTIMES, test_nest2, 1, 1, 1);
  run_ntimes(NTIMES, test_nest2, 4, 2, 2);
  run_ntimes(NTIMES, test_nest3, 1, 1, 1, 1);
  run_ntimes(NTIMES, test_nest3, 1, 2, 1, 1);
  run_ntimes(NTIMES, test_nest3, 1, 1, 2, 1);
  run_ntimes(NTIMES, test_nest3, 1, 1, 1, 2);
  run_ntimes(NTIMES, test_nest3, 2, 1, 1, 1);
  run_ntimes(NTIMES, test_nest4, 1, 1, 1, 1, 1);
  run_ntimes(NTIMES, test_nest4, 2, 1, 1, 1, 1);
  run_ntimes(NTIMES, test_nest4, 1, 2, 1, 1, 1);
  run_ntimes(NTIMES, test_nest4, 1, 1, 2, 1, 1);
  run_ntimes(NTIMES, test_nest4, 1, 1, 1, 2, 1);
  run_ntimes(NTIMES, test_nest4, 1, 1, 1, 1, 2);
  run_ntimes(NTIMES, test_nest4, 1, 1, 1, 1, 1);
  run_ntimes(NTIMES, test_nest4, 1, 2, 1, 2, 1);

  run_ntimes(NTIMES, test_base_split, 4);
  run_ntimes(NTIMES, test_base_split, 2);

  run_ntimes(NTIMES, test_base_split, 7);

  run_ntimes(NTIMES, test_base_split, 1);
  run_ntimes(NTIMES, test_nest_split, 4, 2);
  run_ntimes(NTIMES, test_nest_split, 2, 1);

  run_ntimes(NTIMES, test_nest_split, 7, 2);
  run_ntimes(NTIMES, test_nest_split, 1, 1);
  run_ntimes(NTIMES, test_nest_split, 1, 4);

  run_ntimes(NTIMES, test_nest2_split, 1, 1, 2);
  run_ntimes(NTIMES, test_nest2_split, 1, 2, 1);
  run_ntimes(NTIMES, test_nest2_split, 2, 2, 1);
  run_ntimes(NTIMES, test_nest2_split, 2, 1, 1);
  run_ntimes(NTIMES, test_nest2_split, 4, 2, 1);
  run_ntimes(NTIMES, test_nest2_split, 4, 2, 2);
  run_ntimes(NTIMES, test_nest2_split, 1, 1, 1);
  run_ntimes(NTIMES, test_nest2_split, 4, 2, 2);

  printf("PASS\n");
  return EXIT_SUCCESS;
}
