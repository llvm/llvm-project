// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t
// RUN: %t | %fcheck-generic --check-prefix=LEGACY
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fopenmp
// RUN: %t | %fcheck-generic --check-prefix=LEGACY
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fgpu-default-stream=per-thread
// RUN: %t | %fcheck-generic --check-prefix=PERTHREAD
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

#include <stdio.h>

__global__ void delayedSetValue(int *Out, int Value) {
  volatile unsigned long long Delay = 0;
  for (unsigned I = 0; I < 1000000; ++I)
    Delay += I;
  if (Delay)
    *Out = Value;
}

__global__ void copyValue(int *In, int *Out) { *Out = *In; }

__global__ void waitThenSetValue(int *Gate, int *Out, int Value) {
  volatile int *VolatileGate = Gate;
  for (unsigned I = 0; I < 100000000 && *VolatileGate == 0; ++I)
    ;
  *Out = Value;
}

__global__ void copyValueAndRelease(int *In, int *Out, int *Gate) {
  *Out = *In;
  volatile int *VolatileGate = Gate;
  *VolatileGate = 1;
}

int main(int argc, char **argv) {
  hipStream_t BlockingStream = nullptr;
  if (hipStreamCreateWithFlags(&BlockingStream, hipStreamDefault) != hipSuccess)
    return 1;
  hipStream_t NonBlockingStream = nullptr;
  if (hipStreamCreateWithFlags(&NonBlockingStream, hipStreamNonBlocking) !=
      hipSuccess)
    return 1;

  int *In = nullptr;
  int *Out = nullptr;
  int *Gate = nullptr;
  if (hipMalloc(&In, sizeof(int)) != hipSuccess)
    return 1;
  if (hipMalloc(&Out, sizeof(int)) != hipSuccess)
    return 1;
  if (hipMalloc(&Gate, sizeof(int)) != hipSuccess)
    return 1;

  int Initial = 0;
  int Result = 0;
  if (hipMemcpy(In, &Initial, sizeof(int), hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  if (hipMemcpy(Out, &Initial, sizeof(int), hipMemcpyHostToDevice) !=
      hipSuccess)
    return 1;

  delayedSetValue<<<1, 1, 0, BlockingStream>>>(In, 99);
  copyValue<<<1, 1>>>(In, Out);
  if (hipMemcpy(&Result, Out, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;

  printf("legacy default waited on blocking stream: %d\n", Result);
  // LEGACY: legacy default waited on blocking stream: 99
  // PERTHREAD: legacy default waited on blocking stream: 0

  Result = 0;
  if (hipMemcpy(Out, &Initial, sizeof(int), hipMemcpyHostToDevice) !=
      hipSuccess)
    return 1;

  delayedSetValue<<<1, 1>>>(In, 123);
  copyValue<<<1, 1, 0, BlockingStream>>>(In, Out);
  if (hipStreamSynchronize(BlockingStream) != hipSuccess)
    return 1;
  if (hipMemcpy(&Result, Out, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;

  printf("blocking stream waited on legacy default: %d\n", Result);
  // LEGACY: blocking stream waited on legacy default: 123
  // PERTHREAD: blocking stream waited on legacy default: 99

  Result = 0;
  if (hipMemcpy(In, &Initial, sizeof(int), hipMemcpyHostToDevice) != hipSuccess)
    return 1;
  if (hipMemcpy(Out, &Initial, sizeof(int), hipMemcpyHostToDevice) !=
      hipSuccess)
    return 1;
  if (hipMemcpy(Gate, &Initial, sizeof(int), hipMemcpyHostToDevice) !=
      hipSuccess)
    return 1;

  waitThenSetValue<<<1, 1>>>(Gate, In, 321);
  copyValueAndRelease<<<1, 1, 0, NonBlockingStream>>>(In, Out, Gate);
  if (hipStreamSynchronize(NonBlockingStream) != hipSuccess)
    return 1;
  if (hipMemcpy(&Result, Out, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;

  printf("nonblocking stream did not wait on legacy default: %d\n", Result);
  // LEGACY: nonblocking stream did not wait on legacy default: 0
  // PERTHREAD: nonblocking stream did not wait on legacy default: 0

  if (hipStreamDestroy(BlockingStream) != hipSuccess)
    return 1;
  if (hipStreamDestroy(NonBlockingStream) != hipSuccess)
    return 1;
  if (hipFree(In) != hipSuccess)
    return 1;
  if (hipFree(Out) != hipSuccess)
    return 1;
  if (hipFree(Gate) != hipSuccess)
    return 1;
}
