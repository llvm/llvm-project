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

static void print_error(const char *Label, cudaError_t Error) {
  printf("%s value: %u\n", Label, static_cast<unsigned>(Error));
  printf("%s name: %s\n", Label, cudaGetErrorName(Error));
}

__global__ void setValue(int *Out, int Value) { *Out = Value; }

int main(int argc, char **argv) {
  cudaStream_t Stream = nullptr;
  if (cudaStreamCreate(&Stream) != cudaSuccess)
    return 1;

  printf("stream created: %d\n", Stream != nullptr);
  // CHECK: stream created: 1

  int *StreamPtr = nullptr;
  int *DefaultPtr = nullptr;
  int StreamResult = 0;
  int DefaultResult = 0;
  if (cudaMalloc(&StreamPtr, sizeof(int)) != cudaSuccess)
    return 1;
  if (cudaMalloc(&DefaultPtr, sizeof(int)) != cudaSuccess)
    return 1;

  setValue<<<1, 1, 0, Stream>>>(StreamPtr, 42);
  setValue<<<1, 1>>>(DefaultPtr, 17);

  if (cudaStreamSynchronize(Stream) != cudaSuccess)
    return 1;
  if (cudaDeviceSynchronize() != cudaSuccess)
    return 1;
  if (cudaMemcpy(&StreamResult, StreamPtr, sizeof(int),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return 1;
  if (cudaMemcpy(&DefaultResult, DefaultPtr, sizeof(int),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return 1;

  printf("stream result: %d\n", StreamResult);
  // CHECK: stream result: 42
  printf("default result: %d\n", DefaultResult);
  // CHECK: default result: 17

  if (cudaStreamDestroy(Stream) != cudaSuccess)
    return 1;
  print_error("destroyed stream destroy", cudaStreamDestroy(Stream));
  // CHECK: destroyed stream destroy value: 4
  // CHECK: destroyed stream destroy name: cudaErrorInvalidResourceHandle

  cudaFree(StreamPtr);
  cudaFree(DefaultPtr);
}
