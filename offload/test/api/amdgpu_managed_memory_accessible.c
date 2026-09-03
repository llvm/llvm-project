// RUN: %libomptarget-compile-run-and-check-amdgcn-amd-amdhsa

// REQUIRES: amdgcn-amd-amdhsa

// Device managed memory is accessible by the device it was allocated for.

#include <omp.h>
#include <stdio.h>

void *llvm_omp_target_alloc_shared(size_t, int);
void llvm_omp_target_free_shared(void *, int);

int main() {
  const size_t Size = 1024;
  const int Device = omp_get_default_device();

  char *Shared = llvm_omp_target_alloc_shared(Size, Device);
  if (!Shared) {
    printf("FAIL: allocation\n");
    return 1;
  }

  int Accessible =
      omp_target_is_accessible(Shared, Size, Device) &&
      omp_target_is_accessible(&Shared[Size / 2], Size / 2, Device);

  llvm_omp_target_free_shared(Shared, Device);

  // CHECK: PASS
  printf(Accessible ? "PASS\n" : "FAIL\n");
}
