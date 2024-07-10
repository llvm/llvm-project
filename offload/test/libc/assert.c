// RUN: %libomptarget-compile-generic && %libomptarget-run-fail-generic 2>&1 | \
// RUN:   %fcheck-generic --check-prefix=CHECK

// REQUIRES: libc

// NVPTX without LTO uses the implementation in OpenMP currently.
// UNSUPPORTED: nvptx64-nvidia-cuda
// REQUIRES: gpu

#include <assert.h>

int main() {
  // CHECK: Assertion failed: '0 && "Trivial failure"' in function: 'int main()'
  // CHECK-NOT: Assertion failed:
#pragma omp target
#pragma omp parallel
  { assert(0 && "Trivial failure"); }
}
