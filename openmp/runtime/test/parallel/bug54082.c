// This test is adapted from test_parallel_for_allocate.c in SOLLVE V&V.
// https://github.com/SOLLVE/sollve_vv/blob/master/tests/5.0/parallel_for/test_parallel_for_allocate.c
// RUN: %libomp-compile-and-run
#include <omp.h>

#include <assert.h>
#include <stdlib.h>

#define N 128

int main(int argc, char *argv[]) {
  int errors = 0;
  int *x;
  int result[N][N];
  int successful_alloc = 0;

  omp_memspace_handle_t x_memspace = omp_default_mem_space;
  omp_alloctrait_t x_traits[1] = {omp_atk_alignment, 64};
  omp_allocator_handle_t x_alloc = omp_init_allocator(x_memspace, 1, x_traits);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      result[i][j] = -1;
    }
  }

#pragma omp parallel for allocate(x_alloc: x) private(x) shared(result)
  for (int i = 0; i < N; i++) {
    x = (int *)malloc(N * sizeof(int));
    if (x != NULL) {
#pragma omp simd simdlen(16) aligned(x : 64)
      for (int j = 0; j < N; j++) {
        x[j] = j * i;
      }
      for (int j = 0; j < N; j++) {
        result[i][j] = x[j];
      }
      free(x);
      successful_alloc++;
    }
  }

  errors += successful_alloc < 1;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      errors += result[i][j] != i * j;
    }
  }

  omp_destroy_allocator(x_alloc);

  return errors;
}
