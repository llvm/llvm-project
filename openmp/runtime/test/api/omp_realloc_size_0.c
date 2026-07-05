// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <omp.h>

int main()
{
  omp_alloctrait_t at[2];
  omp_allocator_handle_t a;
  omp_allocator_handle_t f_a;
  void *ptr[2];
  void *nptr[2];
  at[0].key = omp_atk_pool_size;
  at[0].value = 2*1024*1024;
  at[1].key = omp_atk_fallback;
  at[1].value = omp_atv_default_mem_fb;

  a = omp_init_allocator(omp_large_cap_mem_space, 2, at);
  f_a = omp_init_allocator(omp_default_mem_space, 2, at);
  printf("allocator large created: %p\n", (void *)a);
  printf("allocator default created: %p\n", (void *)f_a);

  #pragma omp parallel num_threads(2)
  {
    int i = omp_get_thread_num();
    ptr[i] = omp_alloc(1024 * 1024, f_a);
    #pragma omp barrier
    nptr[i] = omp_realloc(ptr[i], 0, a, f_a);
    #pragma omp barrier
    printf("th %d, nptr %p\n", i, nptr[i]);
    omp_free(nptr[i], a);
  }

  // Both ptr pointers should be non-NULL
  if (ptr[0] == NULL || ptr[1] == NULL) {
    printf("failed: pointers %p %p\n", ptr[0], ptr[1]);
    return 1;
  }
  // Both nptr pointers should be NULL
  if (nptr[0] != NULL || nptr[1] != NULL) {
    printf("failed: pointers %p %p\n", nptr[0], nptr[1]);
    return 1;
  }
  printf("passed\n");
  return 0;
}
