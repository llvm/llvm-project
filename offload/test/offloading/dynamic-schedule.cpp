// clang-format off
// RUN: %libomptarget-compilexx-generic && %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

// REQUIRES: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>

#define N 100

bool schedule(int lb, int ub, int stride, int chunk) {
  int i;

  int result[N];
  for (i = 0; i < N; i++) {
    result[i] = 0;
  }

#pragma omp target parallel for schedule(dynamic, chunk)                       \
    map(tofrom : result[ : N])
  for (i = lb; i < ub; i += stride) {
    result[i] += i;
  }

  int value = 0;
  bool success = true;
  for (i = 0; i < N; i += stride) {
    if (value != result[i]) {
      printf("ERROR: result[%d] = %d instead of %d\n", i, result[i], value);
      success = false;
      break;
    }
    value += stride;
  }

  return success;
}

int main() {
  // CHECK: SUCCESS CHUNK SIZE 1
  if (schedule(0, N, 5, 1))
    printf("SUCCESS CHUNK SIZE 1\n");

  // CHECK: SUCCESS CHUNK SIZE 3
  if (schedule(0, N, 5, 3))
    printf("SUCCESS CHUNK SIZE 3\n");

  return 0;
}
