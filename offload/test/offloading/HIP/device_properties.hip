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
  hipDeviceProp_t Prop = {};
  hipError_t Err = hipGetDeviceProperties(&Prop, 0);
  if (Err != hipSuccess) {
    printf("hipGetDeviceProperties failed: %u\n", Err);
    return 1;
  }

  printf("Device name: %s\n", Prop.name);
  // CHECK: Device name:
  printf("Total global memory: %zu\n", Prop.totalGlobalMem);
  // CHECK: Total global memory:
  printf("Multiprocessors: %i\n", Prop.multiProcessorCount);
  // CHECK: Multiprocessors:
  printf("Warp size: %i\n", Prop.warpSize);
  // CHECK: Warp size:

  if (!Prop.name[0] || !Prop.totalGlobalMem || !Prop.multiProcessorCount ||
      !Prop.warpSize)
    return 1;

  printf("Device properties are populated.\n");
  // CHECK: Device properties are populated.
}
