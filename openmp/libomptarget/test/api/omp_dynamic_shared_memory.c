// RUN: %libomptarget-compile-nvptx64-nvidia-cuda
// RUN: env LIBOMPTARGET_SHARED_MEMORY_SIZE=256 \
// RUN:   %libomptarget-run-nvptx64-nvidia-cuda | %fcheck-nvptx64-nvidia-cuda
// REQUIRES: nvptx64-nvidia-cuda

#include <omp.h>
#include <stdio.h>

int main() {
  int x;
#pragma omp target parallel map(from : x)
  {
    int *buf = llvm_omp_target_dynamic_shared_alloc() + 252;
#pragma omp barrier
    if (omp_get_thread_num() == 0)
      *buf = 1;
#pragma omp barrier
    if (omp_get_thread_num() == 1)
      x = *buf;
  }

  // CHECK: PASS
  if (x == 1 && llvm_omp_target_dynamic_shared_alloc() == NULL)
    printf("PASS\n");
}
