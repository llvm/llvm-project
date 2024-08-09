
// clang-format off
// RUN: %libomptarget-compile-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,TRACE,NDEBG,NOSAN
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,NOSAN
// RUN: %libomptarget-compile-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,TRACE,DEBUG,NOSAN
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,NOSAN
// RUN: %libomptarget-compile-generic -g -mllvm -amdgpu-enable-offload-sanitizer
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=1 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,SANIT,TRACE,DEBUG
// RUN: %not --crash %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,SANIT
// clang-format on

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO
// UNSUPPORTED: s390x-ibm-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu-LTO

#include <omp.h>

int main(void) {

#pragma omp target
  {
  }
#pragma omp target
  {}
#pragma omp target teams num_teams(32) thread_limit(128)
  {
#pragma omp parallel
    if (omp_get_team_num() == 17 && omp_get_thread_num() == 42)
      __builtin_trap();
  }
#pragma omp target
  {
  }
}
// clang-format off
// CHECK: OFFLOAD ERROR: Kernel 'omp target in main @ 32 (__omp_offloading_{{.*}}_main_l32)'
// NOSAN: OFFLOAD ERROR: execution stopped, reason is unknown
// NOSAN: Compile with '-mllvm -amdgpu-enable-offload-sanitizer' improved diagnosis 
// SANIT: OFFLOAD ERROR: execution interrupted by hardware trap instruction
// SANIT: Triggered by thread <42,0,0> block <17,0,0> PC 0x{{.*}}
// TRACE:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_trap.c:
// clang-format on
