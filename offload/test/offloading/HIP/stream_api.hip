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

static void print_error(const char *Label, hipError_t Error) {
  printf("%s value: %u\n", Label, static_cast<unsigned>(Error));
  printf("%s name: %s\n", Label, hipGetErrorName(Error));
}

__global__ void setValue(int *Out, int Value) { *Out = Value; }

int main(int argc, char **argv) {
  hipStream_t Stream = nullptr;
  if (hipStreamCreate(&Stream) != hipSuccess)
    return 1;

  printf("stream created: %d\n", Stream != nullptr);
  // CHECK: stream created: 1

  int *StreamPtr = nullptr;
  int *DefaultPtr = nullptr;
  int StreamResult = 0;
  int DefaultResult = 0;
  if (hipMalloc(&StreamPtr, sizeof(int)) != hipSuccess)
    return 1;
  if (hipMalloc(&DefaultPtr, sizeof(int)) != hipSuccess)
    return 1;

  setValue<<<1, 1, 0, Stream>>>(StreamPtr, 42);
  setValue<<<1, 1>>>(DefaultPtr, 17);

  if (hipStreamSynchronize(Stream) != hipSuccess)
    return 1;
  if (hipDeviceSynchronize() != hipSuccess)
    return 1;
  if (hipMemcpy(&StreamResult, StreamPtr, sizeof(int), hipMemcpyDeviceToHost) !=
      hipSuccess)
    return 1;
  if (hipMemcpy(&DefaultResult, DefaultPtr, sizeof(int),
                hipMemcpyDeviceToHost) != hipSuccess)
    return 1;

  printf("stream result: %d\n", StreamResult);
  // CHECK: stream result: 42
  printf("default result: %d\n", DefaultResult);
  // CHECK: default result: 17

  if (hipStreamDestroy(Stream) != hipSuccess)
    return 1;
  print_error("destroyed stream destroy", hipStreamDestroy(Stream));
  // CHECK: destroyed stream destroy value: 4
  // CHECK: destroyed stream destroy name: hipErrorInvalidResourceHandle

  hipFree(StreamPtr);
  hipFree(DefaultPtr);
}
