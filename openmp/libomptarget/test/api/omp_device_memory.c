// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

int main() {
  const int N = 64;

  int *device_ptr =
      omp_alloc(N * sizeof(int), llvm_omp_target_device_mem_alloc);

#pragma omp target teams distribute parallel for is_device_ptr(device_ptr)
  for (int i = 0; i < N; ++i) {
    device_ptr[i] = 1;
  }

  int sum = 0;
#pragma omp target reduction(+ : sum) is_device_ptr(device_ptr)
  for (int i = 0; i < N; ++i)
    sum += device_ptr[i];

  // CHECK: PASS
  if (sum == N)
    printf("PASS\n");

  omp_free(device_ptr, llvm_omp_target_device_mem_alloc);
}
