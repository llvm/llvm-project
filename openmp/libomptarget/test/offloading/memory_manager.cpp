// RUN: %libomptarget-compilexx-run-and-check-generic

// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-oldDriver
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include <omp.h>

#include <cassert>
#include <iostream>

int main(int argc, char *argv[]) {
#pragma omp parallel for
  for (int i = 0; i < 16; ++i) {
    for (int n = 1; n < (1 << 13); n <<= 1) {
      void *p = omp_target_alloc(n * sizeof(int), 0);
      omp_target_free(p, 0);
    }
  }

#pragma omp parallel for
  for (int i = 0; i < 16; ++i) {
    for (int n = 1; n < (1 << 13); n <<= 1) {
      int *p = (int *)omp_target_alloc(n * sizeof(int), 0);
#pragma omp target teams distribute parallel for is_device_ptr(p)
      for (int j = 0; j < n; ++j) {
        p[j] = i;
      }
      int buffer[n];
#pragma omp target teams distribute parallel for is_device_ptr(p)              \
    map(from : buffer)
      for (int j = 0; j < n; ++j) {
        buffer[j] = p[j];
      }
      for (int j = 0; j < n; ++j) {
        assert(buffer[j] == i);
      }
      omp_target_free(p, 0);
    }
  }

  std::cout << "PASS\n";
  return 0;
}

// CHECK: PASS
