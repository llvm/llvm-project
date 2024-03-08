// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp_testsuite.h"

#ifdef _MSC_VER
#define EFFICIENCY_CHECK
#endif

int* Alloc(unsigned size) { 
  int* p = malloc(size * sizeof(int));
  memset(p, 0, size * sizeof(int));
  return p;
}

#define I(i,j) (i * (upper_bound + outer_eq) + j)
char *loop_type[] = {"< ,<", "< ,<=", "<=,< ", "<=,<="};

int test_omp_for_schedule_static_collapse_lower_triangle(unsigned num_threads, unsigned upper_bound, int outer_eq, int inner_eq) {

  unsigned outer_iterations = upper_bound + outer_eq;
  unsigned first_inner_element = inner_eq;
  unsigned last_inner_element = first_inner_element + (outer_iterations - 1);
  unsigned total_iterations =
      (first_inner_element + last_inner_element) * outer_iterations / 2;
  int i, j;

  /* To ensure Correctness, only valid iterations are executed and are executed
     only once. Stores the number of times an iteration is executed. */
  int *execution_count = Alloc(outer_iterations * outer_iterations);
  /* Stores the number of iterations executed by each thread. */
  int* iterations_per_thread = Alloc(num_threads);

  char *loop_type[] = {"< ,<", "< ,<=", "<=,< ", "<=,<="};

  omp_set_num_threads(num_threads);

#ifdef VERBOSE
  fprintf(stderr, "INFO: Using %6d threads for %6d outer iterations (%6d chunks) loop type lower triangle: %s - ", num_threads, upper_bound, total_iterations, loop_type[outer_eq*2 + inner_eq]);
#endif  

#pragma omp parallel shared(iterations_per_thread, execution_count)
  { /* begin of parallel */
    /* Lower triangular execution_count matrix */
    #pragma omp for schedule (static) collapse(2)
    for(i = 0; i < upper_bound + outer_eq; i++) {
      for (j = 0; j < i + inner_eq; j++) {
        iterations_per_thread[omp_get_thread_num()]++;
        execution_count[I(i, j)]++;
      }
    }/* end of for*/
  }/* end of parallel */

  /* check the execution_count array */
  for (i = 0; i < upper_bound + outer_eq; i++) {
    for (j = 0; j < i + inner_eq; j++) {
      /* iteration with j<=i are valid, but should be executed only once */
      if (execution_count[I(i, j)] != 1) {
#ifdef VERBOSE
        fprintf(
            stderr,
            "ERROR: valid iteration [%i,%i]:%i not executed exactly once.\n", i,
            j, execution_count[I(i, j)]);
#endif            
        return 0;
      }
    }
    for (j = i + inner_eq; j < upper_bound + outer_eq; j++) {
      /* iteration with j>=i are invalid should not have executed */
      if (execution_count[I(i, j)] > 0) {
#ifdef VERBOSE
        fprintf(stderr, "ERROR: invalid iteration [%i,%i]:%i executed.\n", i,
                j, execution_count[I(i, j)]);
#endif                
        return 0;
      }
    }
  }
#ifdef EFFICIENCY_CHECK
  /* Ensure the number of iterations executed by each thread is within bounds */
  for(i = 0;i < num_threads; i++) {
    if (iterations_per_thread[i] < total_iterations / num_threads ||
        iterations_per_thread[i] > total_iterations / num_threads + 1) {
#ifdef VERBOSE
      fprintf(stderr, "ERROR: Inefficient Collapse thread:%i [%i,%i]:%i\n", i,
              total_iterations / num_threads,
              total_iterations / num_threads + 1, iterations_per_thread[i]);
#endif                
      return 0;
    }
  }
#endif  
#ifdef VERBOSE
  fprintf(stderr, "PASSED\r");
#endif
  
  free(execution_count);
  free(iterations_per_thread);
  return 1;
}

int test_omp_for_schedule_static_collapse_upper_triangle(unsigned num_threads, unsigned upper_bound) {

  int outer_eq = 0;
  int inner_eq = 0;
  unsigned outer_iterations = upper_bound + outer_eq;
  unsigned last_inner_element = inner_eq;
  unsigned first_inner_element = last_inner_element + outer_iterations + 1;
  unsigned total_iterations =
      (first_inner_element + last_inner_element) * outer_iterations / 2;
  int i, j;

  /* To ensure Correctness, only valid iterations are executed and are executed
     only once. Stores the number of times an iteration is executed. */
  int *execution_count = Alloc(outer_iterations * outer_iterations);
  /* Stores the number of iterations executed by each thread. */
  int* iterations_per_thread = Alloc(num_threads);

  omp_set_num_threads(num_threads);
#ifdef VERBOSE
  fprintf(stderr, "INFO: Using %6d threads for %6d outer iterations (%6d chunks) loop type upper triangle: %s - ", num_threads, upper_bound, total_iterations, loop_type[outer_eq*2 + inner_eq]);
#endif

#pragma omp parallel shared(iterations_per_thread, execution_count)
  { /* begin of parallel */
    /* Lower triangular execution_count matrix */
    #pragma omp for schedule (static) collapse(2)
    for(i = 0; i < upper_bound + outer_eq; i++) {
      for (j = i; j < upper_bound + inner_eq; j++) {
        iterations_per_thread[omp_get_thread_num()]++;
        execution_count[I(i, j)]++;
      }
    }/* end of for*/
  }/* end of parallel */

  /* check the execution_count array */
  for (i = 0; i < upper_bound + outer_eq; i++) {
    for (j = i; j < upper_bound + inner_eq; j++) {
      /* iteration with j>=i are valid, but should be executed only once */
      if (execution_count[I(i, j)] != 1) {
#ifdef VERBOSE
        fprintf(
            stderr,
            "ERROR: valid iteration [%i,%i]:%i not executed exactly once.\n", i,
            j, execution_count[I(i, j)]);
#endif            
        return 0;
      }
    }
    for (j = 0; j < i; j++) {
      /* iteration with j<i are invalid should not have executed */
      if (execution_count[I(i, j)] > 0) {
#ifdef VERBOSE
        fprintf(stderr, "ERROR: invalid iteration [%i,%i]:%i executed.\n", i,
                j, execution_count[I(i, j)]);
#endif                
        return 0;
      }
    }
  }
#ifdef EFFICIENCY_CHECK
  /* Ensure the number of iterations executed by each thread is within bounds */
  for(i = 0;i < num_threads; i++) {
    if (iterations_per_thread[i] < total_iterations / num_threads ||
        iterations_per_thread[i] > total_iterations / num_threads + 1) {
#ifdef VERBOSE
      fprintf(stderr, "ERROR: Inefficient Collapse thread:%i [%i,%i]:%i\n", i,
              total_iterations / num_threads,
              total_iterations / num_threads + 1, iterations_per_thread[i]);
#endif
      return 0;
    }
  }
#endif  
#ifdef VERBOSE
  fprintf(stderr, "PASSED\r");
#endif

  free(execution_count);
  free(iterations_per_thread);
  return 1;
}


int main(int narg, char* argv[]) {
  unsigned min_threads = 0;
  unsigned max_threads = 64;
  unsigned min_iter = 0;
  unsigned max_iter = 64;
  int i, j, outer_eq, inner_eq;

  for(i = min_threads; i <= max_threads; i++)
    for (j = min_iter; j <= max_iter; j++) {
      for (outer_eq = 0;outer_eq <= 1; outer_eq++)
        for (inner_eq = 0; inner_eq <= 1; inner_eq++)
          if (!test_omp_for_schedule_static_collapse_lower_triangle(i, j, outer_eq, inner_eq))
             return 1;
      if (!test_omp_for_schedule_static_collapse_upper_triangle(i, j))
        return 1;
  }
  return 0;
}
