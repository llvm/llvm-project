// RUN: %libomp-compile && env OMP_NUM_THREADS=2,2,2,2,2 OMP_THREAD_LIMIT=16 \
// RUN: %libomp-run
#include <stdio.h>
#include "omp_testsuite.h"

// When compiler supports num_threads clause list format and strict modifier,
// remove the following and use num_threads clause directly
#if defined(__cplusplus)
extern "C" {
#endif

int __kmpc_global_thread_num(void *loc);
void __kmpc_push_num_threads_list(void *loc, int gtid, unsigned length,
                                  int *list);
void __kmpc_push_num_threads_strict(void *loc, int gtid, int nth, int sev,
                                    const char *msg);
void __kmpc_push_num_threads_list_strict(void *loc, int gtid, unsigned length,
                                         int *list, int sev, const char *msg);

#if defined(__cplusplus)
}
#endif

int test_omp_parallel_num_threads_strict() {
  int num_failed = 0;

// Test regular runtime warning about exceeding thread limit.
// Tolerate whatever value was given.
#pragma omp parallel reduction(+ : num_failed) num_threads(22)
#pragma omp single
  num_failed = num_failed + !(omp_get_num_threads() <= 22);

  // Test with 4 threads and strict -- no problem, no warning.
  __kmpc_push_num_threads_strict(NULL, __kmpc_global_thread_num(NULL), 4, 1,
                                 "This warning shouldn't happen.");
#pragma omp parallel reduction(+ : num_failed) // num_threads(strict:4)
#pragma omp single
  num_failed = num_failed + !(omp_get_num_threads() == 4);

  // Exceed limit, specify user warning message. Tolerate whatever was given.
  __kmpc_push_num_threads_strict(NULL, __kmpc_global_thread_num(NULL), 20, 1,
                                 "User-supplied warning for strict.");
#pragma omp parallel reduction(+ : num_failed)
  // num_threads(strict:20) severity(warning)
  // message("User-supplied warning for strict.")
#pragma omp single
  num_failed = num_failed + !(omp_get_num_threads() <= 20);

  // Exceed limit, no user message, use runtime default message for strict.
  // Tolerate whatever value was given.
  __kmpc_push_num_threads_strict(NULL, __kmpc_global_thread_num(NULL), 21, 1,
                                 NULL);
#pragma omp parallel reduction(+ : num_failed) // num_threads(strict:21)
#pragma omp single
  num_failed = num_failed + !(omp_get_num_threads() <= 21);

  // Exceed limit at top level. Should see user warning message.
  int threads3[2] = {24, 2};
  __kmpc_push_num_threads_list_strict(NULL, __kmpc_global_thread_num(NULL), 2,
                                      threads3, 1,
                                      "User-supplied warning on strict list.");
#pragma omp parallel reduction(+ : num_failed)
  // num_threads(strict:24,2)  severity(warning)
  // message("User-supplied warning on strict. list") // 1st level
  {
#pragma omp single
    num_failed = num_failed + !(omp_get_num_threads() <= 24);
#pragma omp parallel reduction(+ : num_failed) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() <= 2);
    }
  }

  // No strict limit at top level. Regular runtime limiting applies.
  __kmpc_push_num_threads_list(NULL, __kmpc_global_thread_num(NULL), 2,
                               threads3);
#pragma omp parallel reduction(+ : num_failed)
  // num_threads(24,2) // 1st level
  {
#pragma omp single
    num_failed = num_failed + !(omp_get_num_threads() <= 24);
#pragma omp parallel reduction(+ : num_failed) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() <= 2);
    }
  }

  return (!num_failed);
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_parallel_num_threads_strict()) {
      num_failed++;
    }
  }
  return num_failed;
}
