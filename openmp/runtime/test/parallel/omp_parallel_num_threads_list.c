// RUN: %libomp-compile && env OMP_NUM_THREADS=2,2,2,2,2 %libomp-run
#include <stdio.h>
#include "omp_testsuite.h"

// When compiler supports num_threads clause list format, remove the following
// and use num_threads clause directly
#if defined(__cplusplus)
extern "C" {
#endif

int __kmpc_global_thread_num(void *loc);
void __kmpc_push_num_threads_list(void *loc, int gtid, unsigned length,
                                  int *list);

#if defined(__cplusplus)
}
#endif

int test_omp_parallel_num_threads_list() {
  int num_failed = 0;

// Initially, 5 levels specified via OMP_NUM_THREADS with 2 threads per level
// Check top 3 levels
#pragma omp parallel reduction(+ : num_failed) // 1st level
  {
#pragma omp single
    num_failed = num_failed + !(omp_get_num_threads() == 2);
#pragma omp parallel reduction(+ : num_failed) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() == 2);
#pragma omp parallel reduction(+ : num_failed) // 3rd level
      {
#pragma omp single
        num_failed = num_failed + !(omp_get_num_threads() == 2);
      } // end 3rd level parallel
    } // end 2nd level parallel
  } // end 1st level parallel

// Make sure that basic single element num_threads clause works
#pragma omp parallel reduction(+ : num_failed) num_threads(4) // 1st level
  {
#pragma omp single
    num_failed = num_failed + !(omp_get_num_threads() == 4);
#pragma omp parallel reduction(+ : num_failed) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() == 2); // Unaffected
#pragma omp parallel reduction(+ : num_failed) // 3rd level
      {
#pragma omp single
        num_failed = num_failed + !(omp_get_num_threads() == 2); // Unaffected
      } // end 3rd level parallel
    } // end 2nd level parallel
  } // end 1st level parallel

// Check that basic single element num_threads clause works on second level
#pragma omp parallel reduction(+ : num_failed) // 1st level
  {
#pragma omp single
    num_failed = num_failed + !(omp_get_num_threads() == 2); // Unaffected
#pragma omp parallel reduction(+ : num_failed) num_threads(4) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() == 4);
#pragma omp parallel reduction(+ : num_failed) // 3rd level
      {
#pragma omp single
        num_failed = num_failed + !(omp_get_num_threads() == 2); // Unaffected
      } // end 3rd level parallel
    } // end 2nd level parallel
  } // end 1st level parallel

  // Try a short list. It should completely overwrite the old settings.
  // We need to use the compiler interface for now.
  int threads[2] = {3, 3};
  __kmpc_push_num_threads_list(NULL, __kmpc_global_thread_num(NULL), 2,
                               threads);
#pragma omp parallel reduction(+ : num_failed) // num_threads(3,3) // 1st level
  {
#pragma omp single
    num_failed = num_failed + !(omp_get_num_threads() == 3);
#pragma omp parallel reduction(+ : num_failed) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() == 3);
#pragma omp parallel reduction(+ : num_failed) // 3rd level
      {
// NOTE: should just keep using last element in list, to nesting depth
#pragma omp single
        num_failed = num_failed + !(omp_get_num_threads() == 3);
      } // end 3rd level parallel
    } // end 2nd level parallel
  } // end 1st level parallel

// Similar,  but at a lower level.
#pragma omp parallel reduction(+ : num_failed) // 1st level
  {
#pragma omp single
    num_failed = num_failed + !(omp_get_num_threads() == 2); // Unaffected
    int threads[2] = {3, 3};
    __kmpc_push_num_threads_list(NULL, __kmpc_global_thread_num(NULL), 2,
                                 threads);
#pragma omp parallel reduction(+ : num_failed) // num_threads(3,3) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() == 3);
#pragma omp parallel reduction(+ : num_failed) // 3rd level
      {
// NOTE: just keep using last element in list, to nesting depth
#pragma omp single
        num_failed = num_failed + !(omp_get_num_threads() == 3);
      } // end 3rd level parallel
    } // end 2nd level parallel
// Make sure a second inner parallel is NOT affected by the clause
#pragma omp parallel reduction(+ : num_failed) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() == 2); // Unaffected
#pragma omp parallel reduction(+ : num_failed) // 3rd level
      {
#pragma omp single
        // NOTE: just keep using last element in list, to nesting depth
        num_failed = num_failed + !(omp_get_num_threads() == 2); // Unaffected
      } // end 3rd level parallel
    } // end 2nd level parallel
  } // end 1st level parallel

  // Test lists at multiple levels
  int threads2[2] = {3,2};
  __kmpc_push_num_threads_list(NULL, __kmpc_global_thread_num(NULL), 2,
                               threads2);
#pragma omp parallel reduction(+ : num_failed) // num_threads(3,2) // 1st level
  {
#pragma omp single
    num_failed = num_failed + !(omp_get_num_threads() == 3);
#pragma omp parallel reduction(+ : num_failed) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() == 2);
#pragma omp parallel reduction(+ : num_failed) // 3rd level
      {
#pragma omp single
        num_failed = num_failed + !(omp_get_num_threads() == 2);
        int threads3[2] = {3,1};
        __kmpc_push_num_threads_list(NULL, __kmpc_global_thread_num(NULL), 2,
                                     threads3);
#pragma omp parallel reduction(+ : num_failed) // num_threads(3,1) // 4th level
        {
#pragma omp single
          num_failed = num_failed + !(omp_get_num_threads() == 3);
#pragma omp parallel reduction(+ : num_failed) // 5th level
          {
#pragma omp single
            num_failed = num_failed + !(omp_get_num_threads() == 1);
#pragma omp parallel reduction(+ : num_failed) // 6th level
            {
#pragma omp single
              num_failed = num_failed + !(omp_get_num_threads() == 1);
            } // end 6th level parallel
          } // end 5th level parallel
        } // end 4th level parallel
#pragma omp parallel reduction(+ : num_failed) // 4th level
        {
#pragma omp single
          num_failed = num_failed + !(omp_get_num_threads() == 2);
        } // end 4th level parallel
      } // end 3rd level parallel
    } // end 2nd level parallel
#pragma omp parallel reduction(+ : num_failed) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() == 2);
#pragma omp parallel reduction(+ : num_failed) // 3rd level
      {
#pragma omp single
        num_failed = num_failed + !(omp_get_num_threads() == 2);
      } // end 3rd level parallel
    } // end 2nd level parallel
  } // end 1st level parallel

// Now we should be back to the way we started.
#pragma omp parallel reduction(+ : num_failed) // 1st level
  {
#pragma omp single
    num_failed = num_failed + !(omp_get_num_threads() == 2);
#pragma omp parallel reduction(+ : num_failed) // 2nd level
    {
#pragma omp single
      num_failed = num_failed + !(omp_get_num_threads() == 2);
#pragma omp parallel reduction(+ : num_failed) // 3rd level
      {
#pragma omp single
        num_failed = num_failed + !(omp_get_num_threads() == 2);
      } // end 3rd level parallel
    } // end 2nd level parallel
  } // end 1st level parallel

  return (!num_failed);
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_parallel_num_threads_list()) {
      num_failed++;
    }
  }
  return num_failed;
}
