// RUN: %libomp-compile-and-run

// REQUIRES: openmp-5.0

#include <stdio.h>
#include <stdint.h>
#include <omp.h>
#include "omp_testsuite.h"

#define ARRAY_SIZE 10000

int test_omp_alloc() {
  int err;
  int i, j;
  int *shared_array;
  const omp_allocator_t *allocator;
  const omp_allocator_t *test_allocator;
  // Currently, only default memory allocator is implemented
  const omp_allocator_t *allocators[] = {
      omp_default_mem_alloc,
  };

  err = 0;
  for (i = 0; i < sizeof(allocators) / sizeof(allocators[0]); ++i) {
    allocator = allocators[i];
    printf("Using %p allocator\n", test_allocator);
    omp_set_default_allocator(allocator);
    test_allocator = omp_get_default_allocator();
    if (test_allocator != allocator) {
      printf("error: omp_set|get_default_allocator() not working\n");
      return 0;
    }
    shared_array = (int *)omp_alloc(sizeof(int) * ARRAY_SIZE, test_allocator);
    if (shared_array == NULL) {
      printf("error: shared_array is NULL\n");
      return 0;
    }
    for (j = 0; j < ARRAY_SIZE; ++j) {
      shared_array[j] = j;
    }
    #pragma omp parallel shared(shared_array)
    {
      int i;
      int tid = omp_get_thread_num();
      int *private_array =
          (int *)omp_alloc(sizeof(int) * ARRAY_SIZE, omp_default_mem_alloc);
      if (private_array == NULL) {
        printf("error: thread %d private_array is NULL\n", tid);
        #pragma omp atomic
        err++;
      }
      for (i = 0; i < ARRAY_SIZE; ++i) {
        private_array[i] = shared_array[i] + tid;
      }
      for (i = 0; i < ARRAY_SIZE; ++i) {
        if (private_array[i] != i + tid) {
          printf("error: thread %d element %d is %d instead of %d\n", tid, i,
                 private_array[i], i + tid);
          #pragma omp atomic
          err++;
        }
      }
      omp_free(private_array, omp_default_mem_alloc);
    } /* end of parallel */
    omp_free(shared_array, test_allocator);
  }

  return !err;
}

int main() {
  int i;
  int num_failed = 0;

  for (i = 0; i < REPETITIONS; i++) {
    if (!test_omp_alloc()) {
      num_failed++;
    }
  }
  return num_failed;
}
