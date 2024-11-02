// RUN: %libomp-compile-and-run
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "omp.h"

#ifndef MAX_BOUND
#define MAX_BOUND 64
#endif
#ifndef _MSC_VER
#define NO_EFFICIENCY_CHECK
#endif

/* To ensure Correctness, only valid iterations are executed and are executed
   only once. Stores the number of times an iteration is executed. */
unsigned *execution_count = NULL;
/* Stores the number of iterations executed by each thread. */
unsigned *iterations_per_thread = NULL;

unsigned *Alloc(unsigned bound1, unsigned bound2) {
  return (unsigned *)(malloc(bound1 * bound2 * sizeof(unsigned)));
}

void ZeroOut(unsigned *p, unsigned bound1, unsigned bound2) {
  memset(p, 0, bound1 * bound2 * sizeof(unsigned));
}

void Free(unsigned *p) { free((void *)p); }

unsigned *Index(unsigned *p, unsigned i, unsigned j, unsigned bound2) {
  return &p[i * bound2 + j];
}

int test(unsigned upper_bound) {

  unsigned total_iterations = upper_bound * (upper_bound - 1) / 2;
  unsigned num_threads = omp_get_max_threads();
  unsigned lower_per_chunk = total_iterations / num_threads;
  unsigned upper_per_chunk =
      lower_per_chunk + ((total_iterations % num_threads) ? 1 : 0);
  int i, j;

  omp_set_num_threads(num_threads);

  ZeroOut(execution_count, upper_bound, upper_bound);
  ZeroOut(iterations_per_thread, num_threads, 1);

#ifdef VERBOSE
  fprintf(stderr,
          "INFO: Using %6d threads for %6d outer iterations with %6d [%6d:%6d] "
          "chunks "
          "loop type lower triangle <,< - ",
          num_threads, upper_bound, total_iterations, lower_per_chunk,
          upper_per_chunk);
#endif

#pragma omp parallel shared(iterations_per_thread, execution_count)
  { /* begin of parallel */
    /* Lower triangular execution_count matrix */
#pragma omp for schedule(static) collapse(2)
    for (i = 0; i < upper_bound; i++) {
      for (j = 0; j < i; j++) {
        (*Index(iterations_per_thread, omp_get_thread_num(), 0, 1))++;
        (*Index(execution_count, i, j, upper_bound))++;
      }
    } /* end of for*/
  } /* end of parallel */

  /* check the execution_count array */
  for (i = 0; i < upper_bound; i++) {
    for (j = 0; j < i; j++) {
      unsigned value = *Index(execution_count, i, j, upper_bound);
      /* iteration with j<=i are valid, but should have been executed only once
       */
      if (value != 1) {
        fprintf(stderr, "ERROR: valid iteration [%i,%i] executed %i times.\n",
                i, j, value);
        return 0;
      }
    }
    for (j = i; j < upper_bound; j++) {
      unsigned value = *Index(execution_count, i, j, upper_bound);
      /* iteration with j>=i are invalid and should not have been executed
       */
      if (value > 0) {
        fprintf(stderr, "ERROR: invalid iteration [%i,%i] executed %i times.\n",
                i, j, value);
        return 0;
      }
    }
  }

#ifndef NO_EFFICIENCY_CHECK
  /* Ensure the number of iterations executed by each thread is within bounds */
  for (i = 0; i < num_threads; i++) {
    unsigned value = *Index(iterations_per_thread, i, 0, 1);
    if (value < lower_per_chunk || value > upper_per_chunk) {
      fprintf(stderr,
              "ERROR: Inefficient Collapse thread %d of %d assigned %i "
              "iterations; must be between %d and %d\n",
              i, num_threads, value, lower_per_chunk, upper_per_chunk);
      return 0;
    }
  }
#endif
#ifdef VERBOSE
  fprintf(stderr, "PASSED\r\n");
#endif
  return 1;
}

int main() {

  execution_count = Alloc(MAX_BOUND, MAX_BOUND);
  iterations_per_thread = Alloc(omp_get_max_threads(), 1);

  for (unsigned j = 0; j < MAX_BOUND; j++) {
    if (!test(j))
      return 1;
  }
  Free(execution_count);
  Free(iterations_per_thread);
  return 0;
}
