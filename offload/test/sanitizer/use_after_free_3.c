// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 LIBOMPTARGET_MEMORY_MANAGER_THRESHOLD=1024 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK-PASS
// clang-format on

// If offload memory pooling is enabled for a large allocation, reuse error is
// not detected. Run the test w/ and w/o ENV var override on memory pooling
// threshold. REQUIRES: large_allocation_memory_pool

#include <omp.h>
#include <stdio.h>

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
// CHECK: OFFLOAD ERROR: memory access fault by GPU {{.*}} (agent 0x{{.*}}) at virtual address [[PTR:0x[0-9a-z]*]]. Reasons: {{.*}}
// CHECK: Device pointer [[PTR]] points into prior host-issued allocation:
// CHECK: Last deallocation:
// CHECK: Last allocation of size 1073741824
// clang-format on
#pragma omp target
  {
    *P = 5;
  }

  // CHECK-PASS: PASS
  printf("PASS\n");
  return 0;
}
