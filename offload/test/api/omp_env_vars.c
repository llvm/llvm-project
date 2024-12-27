// RUN: %libomptarget-compile-generic
// RUN: env OMP_NUM_TEAMS=1 OMP_TEAMS_THREAD_LIMIT=1 LIBOMPTARGET_INFO=16 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic
// REQUIRES: gpu

#define N 256

int main() {
  // CHECK: Launching kernel [[KERNEL:.+_main_.+]] with [1,1,1] blocks and [1,1,1] threads
#pragma omp target teams
#pragma omp parallel
  {}
}
