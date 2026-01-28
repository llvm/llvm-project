// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// clang-format on

// FIXME: https://github.com/llvm/llvm-project/issues/161265
// UNSUPPORTED: nvidiagpu
//
// REQUIRES: gpu
// XFAIL: intelgpu

#include <omp.h>

int main() {
  int N = (1 << 30);
  char *A = (char *)malloc(N);
#pragma omp target map(A[ : N])
  { A[N] = 3; }
  // clang-format off
// CHECK: OFFLOAD ERROR: memory access fault by GPU {{.*}} (agent 0x{{.*}}) at virtual address [[PTR:0x[0-9a-z]*]]. Reasons: {{.*}}
// CHECK: Device pointer [[PTR]] does not point into any (current or prior) host-issued allocation.
// CHECK: Closest host-issued allocation (distance 1 byte; might be by page):
// CHECK: Last allocation of size 1073741824
// clang-format on
}
