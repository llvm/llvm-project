// clang-format off
// RUN: %libomptarget-compile-generic -fopenmp-target-multi-device -fopenmp-force-usm
// RUN: env HSA_XNACK=1 OMPX_APU_MAPS=1 LIBOMPTARGET_NUM_MULTI_DEVICES=2 LIBOMPTARGET_KERNEL_TRACE=1 \
// RUN: %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

// REQUIRES: multi_device

#include <omp.h>
#include <stdio.h>

#define N 10000

#pragma omp declare target
double rho[N];
#pragma omp end declare target

int main() {
  // Init RHO:
  for (int i = 0; i < N; i++)
    rho[i] = 1.0;

    // clang-format off
// CHECK: DEVID:  0 SGN:7 {{.*}} tripcount:10000 rpc:0 md:1 md_LB:0 md_UB:4999
// CHECK: DEVID:  1 SGN:7 {{.*}} tripcount:10000 rpc:0 md:1 md_LB:5000 md_UB:9999
    // clang-format on

#pragma omp target teams distribute parallel for
  for (int i = 0; i < N; i++) {
    rho[i] += 2.0;
  }

  // CHECK: rho[10] = 3.000000 rho[9000] = 3.000000
  printf("rho[10] = %f rho[9000] = %f\n", rho[10], rho[9000]);

  bool error = false;
  for (int i = 0; i < N; i++) {
    if (rho[i] != 3) {
      printf("ERROR: rho[%d] = %f\n", i, rho[i]);
      error = true;
      break;
    }
  }

  // CHECK: SUCCESS!
  if (!error)
    printf("SUCCESS!\n");

  return 0;
}
