// RUN: %libomptarget-compile-run-and-check-nvptx64-nvidia-cuda
// REQUIRES: nvptx64-nvidia-cuda

#include <omp.h>
#include <stdio.h>

int main() {
  const int N = 64;

  int *hst_ptr = omp_alloc(N * sizeof(int), llvm_omp_target_host_mem_alloc);

  for (int i = 0; i < N; ++i)
    hst_ptr[i] = 2;

#pragma omp target teams distribute parallel for map(tofrom : hst_ptr [0:N])
  for (int i = 0; i < N; ++i)
    hst_ptr[i] -= 1;

  int sum = 0;
  for (int i = 0; i < N; ++i)
    sum += hst_ptr[i];

  omp_free(hst_ptr, llvm_omp_target_shared_mem_alloc);
  // CHECK: PASS
  if (sum == N)
    printf("PASS\n");
}
