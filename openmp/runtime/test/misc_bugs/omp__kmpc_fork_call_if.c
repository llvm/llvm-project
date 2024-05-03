// RUN: %libomp-compile  -Wno-implicit-function-declaration && %t | FileCheck %s

#include <stdio.h>
#include <omp.h>

// Microtask function for parallel region
void microtask(int *global_tid, int *bound_tid) {
  // CHECK: PASS
  if (omp_in_parallel()) {
    printf("FAIL\n");
  } else {
    printf("PASS\n");
  }
}

int main() {
  // Condition for parallelization (false in this case)
  int cond = 0;
  // Call __kmpc_fork_call_if
  __kmpc_fork_call_if(NULL, 0, microtask, cond, NULL);
  return 0;
}
