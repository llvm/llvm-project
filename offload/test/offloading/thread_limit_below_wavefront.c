// Only generic mode is affected, so this compiles without optimization: at -O1
// and above the kernel becomes Generic-SPMD and the answer is right either way.

// RUN: %libomptarget-compile-generic
// RUN: %libomptarget-run-generic | %fcheck-generic

// REQUIRES: gpu

#include <stdio.h>

int main(void) {
  long count = 0;

#pragma omp target teams distribute num_teams(5) thread_limit(4)               \
    map(tofrom : count)
  for (int team = 0; team < 5; team++) {
#pragma omp parallel for
    for (int i = 0; i < 6; i++) {
#pragma omp atomic
      count += 1;
    }
  }

  printf("count = %ld\n", count);
  return count != 30;
}

// CHECK: count = 30
