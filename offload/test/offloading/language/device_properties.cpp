// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda
// RUN: %t.cuda | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.omp -fopenmp
// RUN: %t.cuda.omp | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip
// RUN: %t.hip | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.omp -fopenmp
// RUN: %t.hip.omp | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

// clang-format off
#include <stdio.h>
#include "Inputs/DefineTestLanguageNames.inc"
// clang-format on

int main(int argc, char **argv) {
  DeviceProp_t Prop = {};
  Error_t Err = GetDeviceProperties(&Prop, 0);
  if (Err != Success) {
    printf("GetDeviceProperties failed: %u\n", Err);
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
