// RUN: %libomp-compile-and-run

#include <stdio.h>
#include <omp.h>

int main() {
  omp_alloctrait_t at[2];
  omp_allocator_handle_t a;
  void *p[2];
  at[0].key = omp_atk_pool_size;
  at[0].value = 2 * 1024 * 1024;
  at[1].key = omp_atk_fallback;
  at[1].value = omp_atv_null_fb;
  a = omp_init_allocator(omp_high_bw_mem_space, 2, at);
  printf("allocator hbw created: %p\n", (void *)a);
#pragma omp parallel num_threads(2)
  {
    int i = omp_get_thread_num();
    p[i] = omp_alloc(1024 * 1024, a);
    #pragma omp barrier
    printf("th %d, ptr %p\n", i, p[i]);
    omp_free(p[i], a);
  }
  if (a != omp_null_allocator) {
    // As an allocator has some small memory overhead
    // exactly one of the two pointers should be NULL
    // because of NULL fallback requested
    if ((p[0] == NULL && p[1] != NULL) || (p[0] != NULL && p[1] == NULL)) {
      printf("passed\n");
      return 0;
    } else {
      printf("failed: pointers %p %p\n", p[0], p[1]);
      return 1;
    }
  } else {
    // NULL allocator should cause default allocations
    if (p[0] != NULL && p[1] != NULL) {
      printf("passed\n");
      return 0;
    } else {
      printf("failed: pointers %p %p\n", p[0], p[1]);
      return 1;
    }
  }
}
