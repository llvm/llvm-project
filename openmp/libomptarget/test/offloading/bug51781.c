// Use the generic state machine.  On some architectures, other threads in the
// main thread's warp must avoid barrier instructions.
//
// RUN: %libomptarget-compile-run-and-check-generic

// SPMDize.  There is no main thread, so there's no issue.
//
// RUN: %libomptarget-compile-generic -O1 -Rpass=openmp-opt > %t.spmd 2>&1
// RUN: %fcheck-nvptx64-nvidia-cuda -check-prefix=SPMD -input-file=%t.spmd
// RUN: %fcheck-amdgcn-amd-amdhsa -check-prefix=SPMD -input-file=%t.spmd
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
//
// SPMD: Transformed generic-mode kernel to SPMD-mode.

// Use the custom state machine, which must avoid the same barrier problem as
// the generic state machine.
//
// RUN: %libomptarget-compile-generic -O1 -Rpass=openmp-opt \
// RUN:   -mllvm -openmp-opt-disable-spmdization > %t.custom 2>&1
// RUN: %fcheck-nvptx64-nvidia-cuda -check-prefix=CUSTOM -input-file=%t.custom
// RUN: %fcheck-amdgcn-amd-amdhsa -check-prefix=CUSTOM -input-file=%t.custom
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
//
// CUSTOM: Rewriting generic-mode kernel with a customized state machine.

#include <stdio.h>
int main() {
  int x = 0, y = 1;
  #pragma omp target teams num_teams(1) map(tofrom:x, y)
  {
    x = 5;
    #pragma omp parallel
    y = 6;
  }
  // CHECK: 5, 6
  printf("%d, %d\n", x, y);
  return 0;
}
