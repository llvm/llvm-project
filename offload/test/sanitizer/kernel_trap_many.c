// clang-format off
// RUN: %libomptarget-compile-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=24 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=TRACE,NDEBG
// RUN: %libomptarget-compile-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=16 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=TRACE,DEBUG
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

  for (int i = 0; i < 10; ++i) {
#pragma omp target
    {
    }
  }
#pragma omp target
  {
    __builtin_trap();
  }
}
// TRACE: OFFLOAD ERROR: Kernel {{.*}} (__omp_offloading_{{.*}}_main_l26)
// TRACE: OFFLOAD ERROR: execution interrupted by hardware trap instruction
// TRACE:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_trap_many.c:
