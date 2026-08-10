// RUN: %libomp-compile -D_GNU_SOURCE
// RUN: env OMP_NUM_THREADS=2,2 KMP_AFFINITY=reset,granularity=thread,compact %libomp-run
// REQUIRES: linux

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "libomp_test_affinity.h"

#define CHECK_EQUAL 0
#define CHECK_NOT_EQUAL 1

void check_primary_thread_affinity(int line, affinity_mask_t *other_aff,
                                   int type) {
  #pragma omp master
  {
    affinity_mask_t *primary_aff = affinity_mask_alloc();
    get_thread_affinity(primary_aff);
    if (type == CHECK_EQUAL && !affinity_mask_equal(primary_aff, other_aff)) {
      fprintf(stderr, "error: line %d: primary affinity was not equal\n", line);
      exit(EXIT_FAILURE);
    } else if (type == CHECK_NOT_EQUAL &&
               affinity_mask_equal(primary_aff, other_aff)) {
      fprintf(stderr, "error: line %d: primary affinity was equal\n", line);
      exit(EXIT_FAILURE);
    }
    affinity_mask_free(primary_aff);
  }
}

#define CHECK_PRIMARY_THREAD_AFFINITY_EQUAL(other_aff)                         \
  check_primary_thread_affinity(__LINE__, other_aff, CHECK_EQUAL)
#define CHECK_PRIMARY_THREAD_AFFINITY_NOT_EQUAL(other_aff)                     \
  check_primary_thread_affinity(__LINE__, other_aff, CHECK_NOT_EQUAL)

int main() {
  int i;
  affinity_mask_t *initial_mask = affinity_mask_alloc();
  get_thread_affinity(initial_mask);

  for (i = 0; i < 10; ++i) {
    #pragma omp parallel
    {
      CHECK_PRIMARY_THREAD_AFFINITY_NOT_EQUAL(initial_mask);
    }
    CHECK_PRIMARY_THREAD_AFFINITY_EQUAL(initial_mask);
  }

  omp_set_max_active_levels(2);
  for (i = 0; i < 10; ++i) {
    #pragma omp parallel
    {
      CHECK_PRIMARY_THREAD_AFFINITY_NOT_EQUAL(initial_mask);

      #pragma omp parallel
      CHECK_PRIMARY_THREAD_AFFINITY_NOT_EQUAL(initial_mask);

      CHECK_PRIMARY_THREAD_AFFINITY_NOT_EQUAL(initial_mask);
    }
    CHECK_PRIMARY_THREAD_AFFINITY_EQUAL(initial_mask);
  }

  affinity_mask_free(initial_mask);
  return EXIT_SUCCESS;
}
