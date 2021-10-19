// RUN: %libomp-compile
// RUN: env OMP_ALLOCATOR=omp_high_bw_mem_alloc %libomp-run
// RUN: env OMP_ALLOCATOR=omp_default_mem_space %libomp-run
// RUN: env OMP_ALLOCATOR=omp_large_cap_mem_space:alignment=16,pinned=true \
// RUN:     %libomp-run
// RUN: env \
// RUN:     OMP_ALLOCATOR=omp_high_bw_mem_space:pool_size=1048576,fallback=allocator_fb,fb_data=omp_low_lat_mem_alloc \
// RUN:     %libomp-run

#include <stdio.h>
#include <omp.h>

int main() {
  void *p[2];
#pragma omp parallel num_threads(2)
  {
    int i = omp_get_thread_num();
    p[i] = omp_alloc(1024 * 1024, omp_get_default_allocator());
#pragma omp barrier
    printf("th %d, ptr %p\n", i, p[i]);
    omp_free(p[i], omp_get_default_allocator());
  }
  // Both pointers should be non-NULL
  if (p[0] != NULL && p[1] != NULL) {
    printf("passed\n");
    return 0;
  } else {
    printf("failed: pointers %p %p\n", p[0], p[1]);
    return 1;
  }
}
