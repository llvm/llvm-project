// Check that LIBOMPTARGET_KERNEL_EXE_TIME reports one trace line per kernel
// launch, carrying a monotonically increasing launch id and the name of the
// kernel that was launched.
//
// RUN: %libomptarget-compile-generic && \
// RUN:   env LIBOMPTARGET_KERNEL_EXE_TIME=1 %libomptarget-run-generic 2>&1 | \
// RUN:   %fcheck-generic
//
// REQUIRES: amdgpu

int main(void) {
  int X = 0;

#pragma omp target map(tofrom : X)
  X = 1;

#pragma omp target map(tofrom : X)
  X += 1;

  return X == 2 ? 0 : 1;
}

// CHECK: device {{[0-9]+}} info: LaunchID: [[#ID:]] TeamsXthrds:({{.*}}) Duration(ns): {{[0-9]+}} n:__omp_offloading_{{.*}}_main_l{{[0-9]+}}
// CHECK: device {{[0-9]+}} info: LaunchID: [[#ID+1]] TeamsXthrds:({{.*}}) Duration(ns): {{[0-9]+}} n:__omp_offloading_{{.*}}_main_l{{[0-9]+}}
