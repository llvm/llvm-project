// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>

int main() {
  int N = (1 << 30);
  char *A = (char *)malloc(N);
  char *P;
#pragma omp target map(A[ : N]) map(from : P)
  {
    P = &A[N / 2];
    *P = 3;
  }
  // clang-format off
// CHECK: OFFLOAD ERROR: Memory access fault by GPU {{.*}} (agent 0x{{.*}}) at virtual address [[PTR:0x[0-9a-z]*]]. Reasons: {{.*}}
// CHECK: Device pointer [[PTR]] points into prior host-issued allocation:
// CHECK: Last deallocation:
// CHECK: Last allocation of size 1073741824
// clang-format on
#pragma omp target
  { *P = 5; }
}
