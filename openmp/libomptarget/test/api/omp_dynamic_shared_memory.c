// RUN: %libomptarget-compile-nvptx64-nvidia-cuda -fopenmp-target-new-runtime
// RUN: env LIBOMPTARGET_SHARED_MEMORY_SIZE=4 \
// RUN:   %libomptarget-run-nvptx64-nvidia-cuda | %fcheck-nvptx64-nvidia-cuda
// REQUIRES: nvptx64-nvidia-cuda

#include <omp.h>
#include <stdio.h>

void *get_dynamic_shared() { return NULL; }
#pragma omp begin declare variant match(device = {arch(nvptx64)})
extern void *__kmpc_get_dynamic_shared();
void *get_dynamic_shared() { return __kmpc_get_dynamic_shared(); }
#pragma omp end declare variant

int main() {
  int x;
#pragma omp target parallel map(from : x)
  {
    int *buf = get_dynamic_shared();
#pragma omp barrier
    if (omp_get_thread_num() == 0)
      *buf = 1;
#pragma omp barrier
    if (omp_get_thread_num() == 1)
      x = *buf;
  }

  // CHECK: PASS
  if (x == 1)
    printf("PASS\n");
}
