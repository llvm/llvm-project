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
  cudaStream_t BlockingStream = nullptr;
  if (cudaStreamCreateWithFlags(&BlockingStream, cudaStreamDefault) !=
      cudaSuccess)
    return 1;
  cudaStream_t NonBlockingStream = nullptr;
  if (cudaStreamCreateWithFlags(&NonBlockingStream, cudaStreamNonBlocking) !=
      cudaSuccess)
    return 1;

  int *In = nullptr;
  int *Out = nullptr;
  int *Gate = nullptr;
  if (cudaMalloc(&In, sizeof(int)) != cudaSuccess)
    return 1;
  if (cudaMalloc(&Out, sizeof(int)) != cudaSuccess)
    return 1;
  if (cudaMalloc(&Gate, sizeof(int)) != cudaSuccess)
    return 1;

  int Initial = 0;
  int Result = 0;
  if (cudaMemcpy(In, &Initial, sizeof(int), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return 1;
  if (cudaMemcpy(Out, &Initial, sizeof(int), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return 1;

  delayedSetValue<<<1, 1, 0, BlockingStream>>>(In, 99);
  copyValue<<<1, 1>>>(In, Out);
  if (cudaMemcpy(&Result, Out, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 1;

  printf("legacy default waited on blocking stream: %d\n", Result);
  // LEGACY: legacy default waited on blocking stream: 99
  // PERTHREAD: legacy default waited on blocking stream: 0

  Result = 0;
  if (cudaMemcpy(Out, &Initial, sizeof(int), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return 1;

  delayedSetValue<<<1, 1>>>(In, 123);
  copyValue<<<1, 1, 0, BlockingStream>>>(In, Out);
  if (cudaStreamSynchronize(BlockingStream) != cudaSuccess)
    return 1;
  if (cudaMemcpy(&Result, Out, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 1;

  printf("blocking stream waited on legacy default: %d\n", Result);
  // LEGACY: blocking stream waited on legacy default: 123
  // PERTHREAD: blocking stream waited on legacy default: 99

  Result = 0;
  if (cudaMemcpy(In, &Initial, sizeof(int), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return 1;
  if (cudaMemcpy(Out, &Initial, sizeof(int), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return 1;
  if (cudaMemcpy(Gate, &Initial, sizeof(int), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return 1;

  waitThenSetValue<<<1, 1>>>(Gate, In, 321);
  copyValueAndRelease<<<1, 1, 0, NonBlockingStream>>>(In, Out, Gate);
  if (cudaStreamSynchronize(NonBlockingStream) != cudaSuccess)
    return 1;
  if (cudaMemcpy(&Result, Out, sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return 1;

  printf("nonblocking stream did not wait on legacy default: %d\n", Result);
  // LEGACY: nonblocking stream did not wait on legacy default: 0
  // PERTHREAD: nonblocking stream did not wait on legacy default: 0

  if (cudaStreamDestroy(BlockingStream) != cudaSuccess)
    return 1;
  if (cudaStreamDestroy(NonBlockingStream) != cudaSuccess)
    return 1;
  if (cudaFree(In) != cudaSuccess)
    return 1;
  if (cudaFree(Out) != cudaSuccess)
    return 1;
  if (cudaFree(Gate) != cudaSuccess)
    return 1;
}
