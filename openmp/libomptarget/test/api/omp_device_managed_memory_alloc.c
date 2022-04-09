// RUN: %libomptarget-compile-run-and-check-nvptx64-nvidia-cuda
// REQUIRES: nvptx64-nvidia-cuda

#include <omp.h>
#include <stdio.h>

int main() {
  const int N = 64;

  // Allocates device managed memory that is shared between the host and device.
  int *shared_ptr =
      omp_alloc(N * sizeof(int), llvm_omp_target_shared_mem_alloc);

#pragma omp target teams distribute parallel for is_device_ptr(shared_ptr)
  for (int i = 0; i < N; ++i) {
    shared_ptr[i] = 1;
  }

  int sum = 0;
  for (int i = 0; i < N; ++i)
    sum += shared_ptr[i];

  // CHECK: PASS
  if (sum == N)
    printf("PASS\n");

  omp_free(shared_ptr, llvm_omp_target_shared_mem_alloc);
}
