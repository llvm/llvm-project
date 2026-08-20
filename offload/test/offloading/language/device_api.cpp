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
  int Count = 0;
  if (GetDeviceCount(&Count) != Success)
    return 1;

  printf("device count: %d\n", Count);
  // CHECK: device count: {{[1-9][0-9]*}}

  int Device = -1;
  if (GetDevice(&Device) != Success)
    return 1;

  printf("device: %d\n", Device);
  // CHECK: device: {{[0-9]+}}

  if (SetDevice(Device) != Success)
    return 1;

  int After = -1;
  if (GetDevice(&After) != Success)
    return 1;

  printf("device after set: %d\n", After);
  // CHECK: device after set: {{[0-9]+}}

  Error_t Err = SetDevice(-1);
  printf("set invalid device: %u\n", Err);
  // CHECK: set invalid device: 2
}
