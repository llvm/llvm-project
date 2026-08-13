// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t.launch_tu.o -c
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda %S/kernel_tu.cu.inc -o %t.kernel_tu.o -c
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native --offload-link %t.launch_tu.o %t.kernel_tu.o -o %t
// RUN: %t | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

#include <stdio.h>

extern __global__ void square(int *A);

int main(int argc, char **argv) {
  int *Ptr;
  cudaMalloc(&Ptr, 4);
  printf("Ptr %p\n", Ptr);
  // CHECK: Ptr [[Ptr:0x.*]]
  square<<<1, 1>>>(Ptr);
  int I;
  cudaDeviceSynchronize();
  cudaMemcpy(&I, Ptr, sizeof(int), cudaMemcpyDeviceToHost);
  printf("I: %i\n", I);
  // CHECK: I: 42
}
