// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,NTRCE
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,TRACE
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>

void *llvm_omp_target_alloc_host(size_t Size, int DeviceNum);
void llvm_omp_target_free_host(void *Ptr, int DeviceNum);

int main() {
  int N = (1 << 30);
  char *A = (char *)llvm_omp_target_alloc_host(N, omp_get_default_device());
  char *P;
#pragma omp target map(from : P)
  {
    P = &A[N / 2];
    *P = 3;
  }
  llvm_omp_target_free_host(A, omp_get_default_device());
  // clang-format off
// CHECK: OFFLOAD ERROR: Memory access fault by GPU {{.*}} (agent 0x{{.*}}) at virtual address [[PTR:0x[0-9a-z]*]]. Reasons: {{.*}}
// NTRCE: Use 'OFFLOAD_TRACK_ALLOCATION_TRACES=true' to track device allocations
// TRACE: Device pointer [[PTR]] points into prior host-issued allocation:
// TRACE: Last deallocation:
// TRACE: Last allocation of size 1073741824
// clang-format on
#pragma omp target
  { *P = 5; }
}
