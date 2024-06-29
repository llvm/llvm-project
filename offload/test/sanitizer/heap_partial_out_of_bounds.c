// clang-format off
// RUN: %libomptarget-compileopt-generic -fsanitize=offload
// RUN: not %libomptarget-run-generic 2> %t.out
// RUN: %fcheck-generic --check-prefixes=CHECK < %t.out
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

// Align lines.

#include "omp.h"
#include <stdint.h>
#include <stdio.h>

int main(int argc, char **argv) {
  int N = argc > 42 ? 1000 : 100;
  double *A =
      (double *)omp_target_alloc(N * sizeof(*A), omp_get_default_device());
  char *C = ((char *)&A[N - 1] + 1);
#pragma omp target is_device_ptr(A, C)
  {
    // CHECK: is located 793 bytes inside of a 800-byte region
    double *D = (double *)C;
    *D = 3.14;
  }
}
