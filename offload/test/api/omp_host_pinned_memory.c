// RUN: %libomptarget-compile-run-and-check-generic

#include <omp.h>
#include <stdio.h>

// Allocate pinned memory on the host
void *llvm_omp_target_alloc_host(size_t, int);
void llvm_omp_target_free_host(void *, int);

void run_test(int hostDev) {
  const int N = 64;
  const int device = omp_get_default_device();
  int host = hostDev;

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
  if (sum == N)
    printf("PASS\n");
}

int main() {
  // CHECK: PASS
  run_test(omp_get_initial_device());
  // CHECK: PASS
  run_test(omp_initial_device);
}
