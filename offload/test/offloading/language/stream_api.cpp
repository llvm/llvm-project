// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda
// RUN: %t.cuda | %fcheck-generic --check-prefixes=CHECK,CUDA
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.omp -fopenmp
// RUN: %t.cuda.omp | %fcheck-generic --check-prefixes=CHECK,CUDA
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip
// RUN: %t.hip | %fcheck-generic --check-prefixes=CHECK,HIP
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.omp -fopenmp
// RUN: %t.hip.omp | %fcheck-generic --check-prefixes=CHECK,HIP
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

static void print_error(const char *Label, Error_t Error) {
  printf("%s value: %u\n", Label, static_cast<unsigned>(Error));
  printf("%s name: %s\n", Label, GetErrorName(Error));
}

__global__ void setValue(int *Out, int Value) { *Out = Value; }

int main(int argc, char **argv) {
  print_error("null stream create", StreamCreate(nullptr));
  // CHECK: null stream create value: 1
  // CUDA: null stream create name: cudaErrorInvalidValue
  // HIP: null stream create name: hipErrorInvalidValue
  print_error("null flags stream create",
              StreamCreateWithFlags(nullptr, StreamDefault));
  // CHECK: null flags stream create value: 1
  // CUDA: null flags stream create name: cudaErrorInvalidValue
  // HIP: null flags stream create name: hipErrorInvalidValue

  Stream_t InvalidFlagsStream = nullptr;
  print_error("invalid stream flags",
              StreamCreateWithFlags(&InvalidFlagsStream, ~0u));
  // CHECK: invalid stream flags value: 1
  // CUDA: invalid stream flags name: cudaErrorInvalidValue
  // HIP: invalid stream flags name: hipErrorInvalidValue
  printf("invalid flags stream: %d\n", InvalidFlagsStream == nullptr);
  // CHECK: invalid flags stream: 1

  Stream_t Stream = nullptr;
  if (StreamCreate(&Stream) != Success)
    return 1;
  Stream_t BlockingStream = nullptr;
  if (StreamCreateWithFlags(&BlockingStream, StreamDefault) != Success)
    return 1;
  Stream_t NonBlockingStream = nullptr;
  if (StreamCreateWithFlags(&NonBlockingStream, StreamNonBlocking) != Success)
    return 1;

  printf("stream created: %d\n", Stream != nullptr);
  // CHECK: stream created: 1
  printf("stream flags created: %d %d\n", BlockingStream != nullptr,
         NonBlockingStream != nullptr);
  // CHECK: stream flags created: 1 1

  int *StreamPtr = nullptr;
  int *DefaultPtr = nullptr;
  int StreamResult = 0;
  int DefaultResult = 0;
  if (Malloc(&StreamPtr, sizeof(int)) != Success)
    return 1;
  if (Malloc(&DefaultPtr, sizeof(int)) != Success)
    return 1;

  setValue<<<1, 1, 0, Stream>>>(StreamPtr, 42);
  setValue<<<1, 1>>>(DefaultPtr, 17);

  if (StreamSynchronize(Stream) != Success)
    return 1;
  if (Memcpy(&StreamResult, StreamPtr, sizeof(int), MemcpyDeviceToHost) !=
      Success)
    return 1;
  if (Memcpy(&DefaultResult, DefaultPtr, sizeof(int), MemcpyDeviceToHost) !=
      Success)
    return 1;

  printf("stream result: %d\n", StreamResult);
  // CHECK: stream result: 42
  printf("default result: %d\n", DefaultResult);
  // CHECK: default result: 17

  if (StreamDestroy(Stream) != Success)
    return 1;
  if (StreamDestroy(BlockingStream) != Success)
    return 1;
  if (StreamDestroy(NonBlockingStream) != Success)
    return 1;
  print_error("destroyed stream destroy", StreamDestroy(Stream));
  // CHECK: destroyed stream destroy value: 4
  // CUDA: destroyed stream destroy name: cudaErrorInvalidResourceHandle
  // HIP: destroyed stream destroy name: hipErrorInvalidResourceHandle
  print_error("destroyed stream synchronize", StreamSynchronize(Stream));
  // CHECK: destroyed stream synchronize value: 4
  // CUDA: destroyed stream synchronize name: cudaErrorInvalidResourceHandle
  // HIP: destroyed stream synchronize name: hipErrorInvalidResourceHandle

  Free(StreamPtr);
  Free(DefaultPtr);
}
