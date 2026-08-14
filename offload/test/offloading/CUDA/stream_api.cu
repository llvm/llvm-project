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

__global__ void setValue(int *Out) { *Out = 42; }

int main(int argc, char **argv) {
  cudaStream_t Stream = nullptr;
  if (cudaStreamCreate(&Stream) != cudaSuccess)
    return 1;

  printf("stream created: %d\n", Stream != nullptr);
  // CHECK: stream created: 1

  int *DevPtr = nullptr;
  int Result = 0;
  if (cudaMalloc(&DevPtr, sizeof(int)) != cudaSuccess)
    return 1;

  setValue<<<1, 1, 0, Stream>>>(DevPtr);

  if (cudaStreamSynchronize(Stream) != cudaSuccess)
    return 1;
  if (cudaMemcpy(&Result, DevPtr, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 1;

  printf("stream result: %d\n", Result);
  // CHECK: stream result: 42

  if (cudaStreamDestroy(Stream) != cudaSuccess)
    return 1;
  cudaFree(DevPtr);
}
