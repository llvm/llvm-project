// RUN: %libomp-compile
// RUN: env KMP_AFFINITY=reset %libomp-run
// RUN: env OMP_NUM_THREADS=4 KMP_HW_SUBSET=:1t                                \
// RUN:     KMP_AFFINITY=reset,granularity=thread,compact %libomp-run
// RUN: env OMP_NUM_THREADS=4 OMP_PLACES='threads(1)' KMP_AFFINITY=reset       \
// RUN:     %libomp-run
// REQUIRES: linux

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "libomp_test_affinity.h"

int main() {
  int a, nth, retval = EXIT_SUCCESS;
  affinity_mask_t *initial_mask = NULL;
  affinity_mask_t *mask_after_parallel = NULL;

  initial_mask = affinity_mask_alloc();
  get_thread_affinity(initial_mask);

  a = 0;
#pragma omp parallel
  {
#pragma omp atomic
    a++;
#pragma omp single nowait
    nth = omp_get_num_threads();
  }
  if (a != nth) {
    fprintf(stderr, "error: a(%d) != nth(%d)\n", a, nth);
    retval = EXIT_FAILURE;
    goto exit_main;
  }

  mask_after_parallel = affinity_mask_alloc();
  get_thread_affinity(mask_after_parallel);

  if (!affinity_mask_equal(mask_after_parallel, initial_mask)) {
    char buf[1024] = {0};
    printf("error: mask after parallel != initial mask ");
    printf("  mask after parallel: ");
    affinity_mask_snprintf(buf, sizeof(buf), mask_after_parallel);
    printf("%s\n", buf);
    printf("  initial mask: ");
    affinity_mask_snprintf(buf, sizeof(buf), initial_mask);
    printf("%s\n", buf);
    retval = EXIT_FAILURE;
    goto exit_main;
  }

exit_main:
  if (initial_mask)
    affinity_mask_free(initial_mask);
  if (mask_after_parallel)
    affinity_mask_free(mask_after_parallel);
  return retval;
}
