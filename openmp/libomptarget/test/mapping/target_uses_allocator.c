// RUN: %libomptarget-compile-run-and-check-generic

// UNSUPPORTED: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>

#define N 1024

int test_omp_aligned_alloc_on_device() {
  int errors = 0;

  omp_memspace_handle_t memspace = omp_default_mem_space;
  omp_alloctrait_t traits[2] = {{omp_atk_alignment, 64}, {omp_atk_access, 64}};
  omp_allocator_handle_t alloc =
      omp_init_allocator(omp_default_mem_space, 1, traits);

#pragma omp target map(tofrom : errors) uses_allocators(alloc(traits))
  {
    int *x;
    int not_correct_array_values = 0;

    x = (int *)omp_aligned_alloc(64, N * sizeof(int), alloc);
    if (x == NULL) {
      errors++;
    } else {
#pragma omp parallel for simd simdlen(16) aligned(x : 64)
      for (int i = 0; i < N; i++) {
        x[i] = i;
      }

#pragma omp parallel for simd simdlen(16) aligned(x : 64)
      for (int i = 0; i < N; i++) {
        if (x[i] != i) {
#pragma omp atomic write
          not_correct_array_values = 1;
        }
      }
      if (not_correct_array_values) {
        errors++;
      }
      omp_free(x, alloc);
    }
  }

  omp_destroy_allocator(alloc);

  return errors;
}

int main() {
  int errors = 0;
  if (test_omp_aligned_alloc_on_device())
    printf("FAILE\n");
  else
    // CHECK: PASSED
    printf("PASSED\n");
}
