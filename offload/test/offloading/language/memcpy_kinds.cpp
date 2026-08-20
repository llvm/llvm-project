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
  int HostSrc = 11;
  int HostDst = 0;
  if (Memcpy(&HostDst, &HostSrc, sizeof(int), MemcpyHostToHost) != Success)
    return 1;

  printf("host to host: %d\n", HostDst);
  // CHECK: host to host: 11

  int *DevSrc = nullptr;
  int *DevDst = nullptr;
  int Result = 0;
  if (Malloc(&DevSrc, sizeof(int)) != Success)
    return 1;
  if (Malloc(&DevDst, sizeof(int)) != Success)
    return 1;

  HostSrc = 42;
  if (Memcpy(DevSrc, &HostSrc, sizeof(int), MemcpyHostToDevice) != Success)
    return 1;
  if (Memcpy(DevDst, DevSrc, sizeof(int), MemcpyDeviceToDevice) != Success)
    return 1;
  if (Memcpy(&Result, DevDst, sizeof(int), MemcpyDeviceToHost) != Success)
    return 1;

  printf("device to device: %d\n", Result);
  // CHECK: device to device: 42

  Free(DevSrc);
  Free(DevDst);
}
