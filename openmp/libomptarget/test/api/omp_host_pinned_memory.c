// RUN: %libomptarget-compile-run-and-check-generic
// UNSUPPORTED: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>

// Allocate pinned memory on the host
void *llvm_omp_target_alloc_host(size_t, int);
void llvm_omp_target_free_host(void *, int);

int main() {
  const int N = 64;
  const int device = omp_get_default_device();
  const int host = omp_get_initial_device();

  int *hst_ptr = llvm_omp_target_alloc_host(N * sizeof(int), device);

  for (int i = 0; i < N; ++i)
    hst_ptr[i] = 2;

#pragma omp target teams distribute parallel for device(device)                \
    map(tofrom : hst_ptr[0 : N])
  for (int i = 0; i < N; ++i)
    hst_ptr[i] -= 1;

  int sum = 0;
  for (int i = 0; i < N; ++i)
    sum += hst_ptr[i];

  llvm_omp_target_free_host(hst_ptr, device);
  // CHECK: PASS
  if (sum == N)
    printf("PASS\n");
}
