
// clang-format off
// RUN: %libomptarget-compile-generic -g -mllvm -amdgpu-enable-offload-sanitizer
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=SANIT
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=SANIT
// RUN: %libomptarget-compileopt-generic -g -mllvm -amdgpu-enable-offload-sanitizer
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=SANIT
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=SANIT

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>

__attribute__((noinline)) void unreachable(volatile int *GoodPtr) {
  *GoodPtr = 1;
  __builtin_unreachable();
}

int main(void) {
#pragma omp target
  {
    volatile int A = 0;
    unreachable(&A);
  }
}
// SANIT: OFFLOAD ERROR: Kernel {{.*}} (__omp_offloading_{{.*}}_main_l27)
// SANIT: OFFLOAD ERROR: execution reached an "unreachable" state (likely caused by undefined behavior)
// SANIT: Triggered by thread <{{.*}},0,0> block <{{.*}},0,0> PC 0x{{.*}}
