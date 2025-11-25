// clang-format off
// RUN: %libomptarget-compile-generic -fopenmp-target-multi-device -fopenmp-target-fast
// RUN: env HSA_XNACK=1 OMPX_APU_MAPS=1 LIBOMPTARGET_KERNEL_TRACE=1 LIBOMPTARGET_NUM_MULTI_DEVICES=2 \
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

#define M 10
#define N 15000

int main() {
  double *a = (double *)malloc(sizeof(double) * (N * M + 1));

  // Init a:
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < N; k++) {
      a[k * M + i] = i;
    }
  }

  double final_sum = 0.0;
#pragma omp target teams distribute reduction(+ : final_sum)
  for (int i = 0; i < M; i++) {
    double sum_qi = 0.0;
#pragma omp parallel for simd reduction(+ : sum_qi)
    for (int k = 0; k < N; k++)
      sum_qi = sum_qi + a[k * M + i];
    final_sum += sum_qi;
  }

  // clang-format off
  // CHECK: DEVID:  0 SGN:3 {{.*}} tripcount:10 rpc:0 md:0
  // clang-format on

  // CHECK: final_sum = 675000
  printf("final_sum = %f\n", final_sum);

  free(a);
  return 0;
}
