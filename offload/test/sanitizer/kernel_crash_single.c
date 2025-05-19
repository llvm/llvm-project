// clang-format off
// RUN: %libomptarget-compile-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=TRACE,NDEBG
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK
// RUN: %libomptarget-compile-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=TRACE,DEBUG
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
  int *A = 0;
#pragma omp target
  {
    *A = 42;
  }
}
// TRACE: Display kernel launch trace
// TRACE: Kernel {{.*}} (__omp_offloading_{{.*}}_main_l23)
// TRACE:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_crash_single.c:23
//
// CHECK: Display only launched kernel:
// CHECK: Kernel {{.*}} (__omp_offloading_{{.*}}_main_l23)
