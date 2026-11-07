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

__global__ void add(int *Ptr, int Value) { *Ptr += Value; }

int main(int argc, char **argv) {
  int *HostAllocPtr = nullptr;
  if (HostAlloc(&HostAllocPtr, sizeof(int), HostAllocDefault) != Success)
    return 1;

  *HostAllocPtr = 17;
  add<<<1, 1>>>(HostAllocPtr, 5);
  if (DeviceSynchronize() != Success)
    return 1;
  printf("HostAlloc value: %d\n", *HostAllocPtr);
  // CHECK: HostAlloc value: 22

  if (FreeHost(HostAllocPtr) != Success)
    return 1;

  int *MallocHostPtr = nullptr;
  if (MallocHost(&MallocHostPtr, sizeof(int)) != Success)
    return 1;

  *MallocHostPtr = 23;
  add<<<1, 1>>>(MallocHostPtr, 7);
  if (DeviceSynchronize() != Success)
    return 1;
  printf("MallocHost value: %d\n", *MallocHostPtr);
  // CHECK: MallocHost value: 30

  if (FreeHost(MallocHostPtr) != Success)
    return 1;
}
