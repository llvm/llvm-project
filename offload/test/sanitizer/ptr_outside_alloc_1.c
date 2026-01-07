// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,NTRCE
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_ALLOCATION_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,TRACE
// clang-format on

// FIXME: https://github.com/llvm/llvm-project/issues/161265
// UNSUPPORTED: nvidiagpu
//
// REQUIRES: gpu

#include <omp.h>

void *llvm_omp_target_alloc_host(size_t Size, int DeviceNum);
void llvm_omp_target_free_host(void *Ptr, int DeviceNum);

int main() {
  int N = (1 << 30);
  char *A = (char *)llvm_omp_target_alloc_host(N, omp_get_default_device());
  char *P;
#pragma omp target map(from : P)
  {
    P = &A[0];
    *P = 3;
  }
  // clang-format off
// CHECK: OFFLOAD ERROR: memory access fault by GPU {{.*}} (agent 0x{{.*}}) at virtual address [[PTR:0x[0-9a-z]*]]. Reasons: {{.*}}
// NTRCE: Use 'OFFLOAD_TRACK_ALLOCATION_TRACES=true' to track device allocations
// TRACE: Device pointer [[PTR]] does not point into any (current or prior) host-issued allocation.
// TRACE: Closest host-issued allocation (distance 4096 bytes; might be by page):
// TRACE: Last allocation of size 1073741824
// clang-format on
#pragma omp target
  { P[-4] = 5; }

  llvm_omp_target_free_host(A, omp_get_default_device());
}
