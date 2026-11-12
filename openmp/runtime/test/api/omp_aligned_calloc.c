// RUN: %libomp-compile-and-run
// UNSUPPORTED: gnu

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#define NTH 8
#define AL0 64
#define AL1 128

int main()
{
  int err = 0;
  omp_alloctrait_t at[3];
  omp_allocator_handle_t a;
  void *p[NTH];
  at[0].key = omp_atk_pool_size;
  at[0].value = 16*1024*1024;
  at[1].key = omp_atk_fallback;
  at[1].value = omp_atv_null_fb;
  a = omp_init_allocator(omp_large_cap_mem_space, 2, at);
  printf("allocator large created: %p\n", (void *)a);
  #pragma omp parallel num_threads(8)
  {
    int i = omp_get_thread_num();
    p[i] = omp_aligned_calloc(AL0, 1024*128, 8, a); // API's alignment only
    #pragma omp barrier
    printf("th %d, ptr %p\n", i, p[i]);
    if ((size_t)p[i] % AL0) {
      #pragma omp atomic
        err++;
      printf("Error param: th %d, ptr %p is not %d-byte aligned\n",
             i, p[i], AL0);
    }
    omp_free(p[i], a);
  }
  omp_destroy_allocator(a);
  at[2].key = omp_atk_alignment;
  at[2].value = AL1;
  a = omp_init_allocator(omp_large_cap_mem_space, 3, at);
  printf("allocator large aligned %d created: %p\n", AL1, (void *)a);
  if (a != omp_null_allocator)
  #pragma omp parallel num_threads(8)
  {
    int i = omp_get_thread_num();
    p[i] = omp_aligned_calloc(AL0, 1024*128, 8, a); // allocator's alignment wins
    #pragma omp barrier
    printf("th %d, ptr %p\n", i, p[i]);
    if ((size_t)p[i] % AL1) {
      #pragma omp atomic
        err++;
      printf("Error allocator: th %d, ptr %p is not %d-byte aligned\n",
             i, p[i], AL1);
    }
    omp_free(p[i], a);
  }
  omp_destroy_allocator(a);
  at[2].key = omp_atk_alignment;
  at[2].value = AL0;
  a = omp_init_allocator(omp_large_cap_mem_space, 3, at);
  printf("allocator large aligned %d created: %p\n", AL0, (void *)a);
  #pragma omp parallel num_threads(8)
  {
    int i = omp_get_thread_num();
    p[i] = omp_aligned_calloc(AL1, 1024*128, 8, a); // API's alignment wins
    #pragma omp barrier
    printf("th %d, ptr %p\n", i, p[i]);
    if ((size_t)p[i] % AL1) {
      #pragma omp atomic
        err++;
      printf("Error param: th %d, ptr %p is not %d-byte aligned\n",
             i, p[i], AL1);
    }
    omp_free(p[i], a);
  }
  omp_destroy_allocator(a);

  if (err == 0) {
    printf("passed\n");
    return 0;
  } else {
    printf("failed\n");
    return 1;
  }
}
