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
  hipStream_t Stream = nullptr;
  if (hipStreamCreate(&Stream) != hipSuccess)
    return 1;

  printf("stream created: %d\n", Stream != nullptr);
  // CHECK: stream created: 1

  int *DevPtr = nullptr;
  int Result = 0;
  if (hipMalloc(&DevPtr, sizeof(int)) != hipSuccess)
    return 1;

  setValue<<<1, 1, 0, Stream>>>(DevPtr);

  if (hipStreamSynchronize(Stream) != hipSuccess)
    return 1;
  if (hipMemcpy(&Result, DevPtr, sizeof(int), hipMemcpyDeviceToHost) !=
      hipSuccess)
    return 1;

  printf("stream result: %d\n", Result);
  // CHECK: stream result: 42

  if (hipStreamDestroy(Stream) != hipSuccess)
    return 1;
  hipFree(DevPtr);
}
