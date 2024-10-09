// clang-format off
// RUN: %libomptarget-compilexx-generic && env LIBOMPTARGET_REUSE_BLOCKS_FOR_HIGH_TRIP_COUNT=False %libomptarget-run-generic 2>&1 | %fcheck-generic
// RUN: %libomptarget-compilexx-generic && %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=DEFAULT

// UNSUPPORTED: aarch64-unknown-linux-gnu 
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO 
// UNSUPPORTED: x86_64-unknown-linux-gnu 
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO 
// UNSUPPORTED: s390x-ibm-linux-gnu 
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO
// clang-format on

/*
  Check if there is a thread for each loop iteration
*/
#include <omp.h>
#include <stdio.h>

int main() {
  int N = 819200;
  int num_threads[N];

#pragma omp target teams distribute parallel for
  for (int j = 0; j < N; j++) {
    num_threads[j] = omp_get_num_threads() * omp_get_num_teams();
  }

  if (num_threads[0] == N)
    // CHECK: PASS
    printf("PASS\n");
  else
    // DEFAULT: FAIL
    printf("FAIL: num_threads: %d\n != N: %d", num_threads[0], N);
  return 0;
}
