// RUN: %libomp-compile && env KMP_TOPOLOGY_METHOD=hwloc %libomp-run
// REQUIRES: hwloc

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
