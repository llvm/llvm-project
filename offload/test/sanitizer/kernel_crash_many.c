// clang-format off
// RUN: %libomptarget-compile-generic
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=24 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,NDEBG
// RUN: %libomptarget-compile-generic -g
// RUN: %not --crash env -u LLVM_DISABLE_SYMBOLIZATION OFFLOAD_TRACK_NUM_KERNEL_LAUNCH_TRACES=16 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefixes=CHECK,DEBUG
// clang-format on

// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: s390x-ibm-linux-gnu
// XFAIL: intelgpu

#include <omp.h>

int main(void) {
  int *A = 0;
  for (int i = 0; i < 10; ++i) {
#pragma omp target
    {
    }
  }
#pragma omp target
  {
    *A = 42;
  }
}
// CHECK: Display 8 of the 8 last kernel launch traces
// CHECK: Kernel 0: {{.*}} (__omp_offloading_{{.*}}_main_l[[@LINE-6]])
// CHECK:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_crash_many.c:[[@LINE-9]]
//
// CHECK: Kernel 1: {{.*}} (__omp_offloading_{{.*}}_main_l[[@LINE-15]])
// CHECK:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_crash_many.c:
//
// CHECK: Kernel 2: {{.*}} (__omp_offloading_{{.*}}_main_l[[@LINE-20]])
// CHECK:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_crash_many.c:
//
// CHECK: Kernel 3: {{.*}} (__omp_offloading_{{.*}}_main_l[[@LINE-25]])
// CHECK:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_crash_many.c:
//
// CHECK: Kernel 4: {{.*}} (__omp_offloading_{{.*}}_main_l[[@LINE-30]])
// CHECK:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_crash_many.c:
//
// CHECK: Kernel 5: {{.*}} (__omp_offloading_{{.*}}_main_l[[@LINE-35]])
// CHECK:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_crash_many.c:
//
// CHECK: Kernel 6: {{.*}} (__omp_offloading_{{.*}}_main_l[[@LINE-40]])
// CHECK:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_crash_many.c:
//
// CHECK: Kernel 7: {{.*}} (__omp_offloading_{{.*}}_main_l[[@LINE-45]])
// CHECK:     launchKernel
// NDEBG:     main
// DEBUG:     main {{.*}}kernel_crash_many.c:
//
// CHECK-NOT: Kernel {{[[0-9]]+}}:
