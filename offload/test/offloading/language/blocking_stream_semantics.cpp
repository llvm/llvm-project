// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.legacy
// RUN: %t.cuda.legacy | %fcheck-generic --check-prefix=LEGACY
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.legacy.omp -fopenmp
// RUN: %t.cuda.legacy.omp | %fcheck-generic --check-prefix=LEGACY
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.perthread -fgpu-default-stream=per-thread
// RUN: %t.cuda.perthread | %fcheck-generic --check-prefix=PERTHREAD
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.legacy
// RUN: %t.hip.legacy | %fcheck-generic --check-prefix=LEGACY
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.legacy.omp -fopenmp
// RUN: %t.hip.legacy.omp | %fcheck-generic --check-prefix=LEGACY
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.perthread -fgpu-default-stream=per-thread
// RUN: %t.hip.perthread | %fcheck-generic --check-prefix=PERTHREAD
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
  Stream_t BlockingStream = nullptr;
  if (StreamCreateWithFlags(&BlockingStream, StreamDefault) != Success)
    return 1;
  Stream_t NonBlockingStream = nullptr;
  if (StreamCreateWithFlags(&NonBlockingStream, StreamNonBlocking) != Success)
    return 1;

  int *In = nullptr;
  int *Out = nullptr;
  int *Gate = nullptr;
  if (Malloc(&In, sizeof(int)) != Success)
    return 1;
  if (Malloc(&Out, sizeof(int)) != Success)
    return 1;
  if (Malloc(&Gate, sizeof(int)) != Success)
    return 1;

  int Initial = 0;
  int Result = 0;
  if (Memcpy(In, &Initial, sizeof(int), MemcpyHostToDevice) != Success)
    return 1;
  if (Memcpy(Out, &Initial, sizeof(int), MemcpyHostToDevice) != Success)
    return 1;

  delayedSetValue<<<1, 1, 0, BlockingStream>>>(In, 99);
  copyValue<<<1, 1>>>(In, Out);
  if (Memcpy(&Result, Out, sizeof(int), MemcpyDeviceToHost) != Success)
    return 1;

  printf("legacy default waited on blocking stream: %d\n", Result);
  // LEGACY: legacy default waited on blocking stream: 99
  // PERTHREAD: legacy default waited on blocking stream: 0

  Result = 0;
  if (Memcpy(Out, &Initial, sizeof(int), MemcpyHostToDevice) != Success)
    return 1;

  delayedSetValue<<<1, 1>>>(In, 123);
  copyValue<<<1, 1, 0, BlockingStream>>>(In, Out);
  if (StreamSynchronize(BlockingStream) != Success)
    return 1;
  if (Memcpy(&Result, Out, sizeof(int), MemcpyDeviceToHost) != Success)
    return 1;

  printf("blocking stream waited on legacy default: %d\n", Result);
  // LEGACY: blocking stream waited on legacy default: 123
  // PERTHREAD: blocking stream waited on legacy default: 99

  Result = 0;
  if (Memcpy(In, &Initial, sizeof(int), MemcpyHostToDevice) != Success)
    return 1;
  if (Memcpy(Out, &Initial, sizeof(int), MemcpyHostToDevice) != Success)
    return 1;
  if (Memcpy(Gate, &Initial, sizeof(int), MemcpyHostToDevice) != Success)
    return 1;

  waitThenSetValue<<<1, 1>>>(Gate, In, 321);
  copyValueAndRelease<<<1, 1, 0, NonBlockingStream>>>(In, Out, Gate);
  if (StreamSynchronize(NonBlockingStream) != Success)
    return 1;
  if (Memcpy(&Result, Out, sizeof(int), MemcpyDeviceToHost) != Success)
    return 1;

  printf("nonblocking stream did not wait on legacy default: %d\n", Result);
  // LEGACY: nonblocking stream did not wait on legacy default: 0
  // PERTHREAD: nonblocking stream did not wait on legacy default: 0

  if (StreamDestroy(BlockingStream) != Success)
    return 1;
  if (StreamDestroy(NonBlockingStream) != Success)
    return 1;
  if (Free(In) != Success)
    return 1;
  if (Free(Out) != Success)
    return 1;
  if (Free(Gate) != Success)
    return 1;
}
