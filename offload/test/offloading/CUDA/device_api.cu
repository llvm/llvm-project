// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t
// RUN: %t | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fopenmp
// RUN: %t | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

#include <stdio.h>

int main(int argc, char **argv) {
  int Count = 0;
  if (cudaGetDeviceCount(&Count) != cudaSuccess)
    return 1;

  printf("device count: %d\n", Count);
  // CHECK: device count: {{[1-9][0-9]*}}

  int Device = -1;
  if (cudaGetDevice(&Device) != cudaSuccess)
    return 1;

  printf("device: %d\n", Device);
  // CHECK: device: {{[0-9]+}}

  if (cudaSetDevice(Device) != cudaSuccess)
    return 1;

  int After = -1;
  if (cudaGetDevice(&After) != cudaSuccess)
    return 1;

  printf("device after set: %d\n", After);
  // CHECK: device after set: {{[0-9]+}}

  cudaError_t Err = cudaSetDevice(-1);
  printf("set invalid device: %u\n", Err);
  // CHECK: set invalid device: 2
}
