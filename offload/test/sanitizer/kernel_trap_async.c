
// clang-format off
// RUN: %libomptarget-compileopt-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,TRACE
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// RUN: %libomptarget-compileopt-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,TRACE,DEBUG
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// clang-format on

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>

int main(void) {

#pragma omp target nowait
  {
  }
#pragma omp target nowait
  {
  }
#pragma omp target nowait
  {
    __builtin_trap();
  }
#pragma omp taskwait
}

// clang-format off
// CHECK: OFFLOAD ERROR: Kernel {{.*}} (__omp_offloading_{{.*}}_main_l30)
// CHECK: OFFLOAD ERROR: execution interrupted by hardware trap instruction
// TRACE:     launchKernel
// DEBUG:     kernel_trap_async.c:
// clang-format on
