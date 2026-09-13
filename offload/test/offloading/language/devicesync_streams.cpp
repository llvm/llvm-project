// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.legacy -fgpu-default-stream=legacy -pthread -std=c++17
// RUN: %t.cuda.legacy | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.perthread -fgpu-default-stream=per-thread -pthread -std=c++17
// RUN: %t.cuda.perthread | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.legacy -fgpu-default-stream=legacy -pthread -std=c++17
// RUN: %t.hip.legacy | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.perthread -fgpu-default-stream=per-thread -pthread -std=c++17
// RUN: %t.hip.perthread | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

// clang-format off
#include <chrono>
#include <cstdio>
#include <thread>
#include "Inputs/DefineTestLanguageNames.inc"
// clang-format on

__global__ void waitThenSet(volatile int *Gate, volatile int *Out, int Value) {
  for (unsigned long long I = 0; I < 1000000000ULL && *Gate == 0; ++I)
    ;
  *Out = *Gate ? Value : -Value;
}

int main(int argc, char **argv) {
  Stream_t BlockingStream = nullptr;
  if (StreamCreateWithFlags(&BlockingStream, StreamDefault) != Success)
    return 1;
  Stream_t NonBlockingStream = nullptr;
  if (StreamCreateWithFlags(&NonBlockingStream, StreamNonBlocking) != Success)
    return 1;

  int *BlockingGate = nullptr;
  int *NonBlockingGate = nullptr;
  int *BlockingOutStorage = nullptr;
  int *NonBlockingOutStorage = nullptr;
  if (HostAlloc(&BlockingGate, sizeof(int), HostAllocDefault) != Success)
    return 1;
  if (HostAlloc(&NonBlockingGate, sizeof(int), HostAllocDefault) != Success)
    return 1;
  if (HostAlloc(&BlockingOutStorage, sizeof(int), HostAllocDefault) != Success)
    return 1;
  if (HostAlloc(&NonBlockingOutStorage, sizeof(int), HostAllocDefault) !=
      Success)
    return 1;

  volatile int *BlockingOut = BlockingOutStorage;
  volatile int *NonBlockingOut = NonBlockingOutStorage;
  *BlockingGate = 0;
  *NonBlockingGate = 0;
  *BlockingOut = 0;
  *NonBlockingOut = 0;

  waitThenSet<<<1, 1, 0, BlockingStream>>>(BlockingGate, BlockingOut, 17);
  waitThenSet<<<1, 1, 0, NonBlockingStream>>>(NonBlockingGate, NonBlockingOut,
                                              23);

  std::thread Releaser([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    *BlockingGate = 1;
    *NonBlockingGate = 1;
  });

  Error_t SyncResult = DeviceSynchronize();

  if (SyncResult == Success) {
    printf("device sync waited on blocking stream: %d\n", *BlockingOut);
    // CHECK: device sync waited on blocking stream: 17
    printf("device sync waited on nonblocking stream: %d\n", *NonBlockingOut);
    // CHECK: device sync waited on nonblocking stream: 23
  }

  Releaser.join();
  if (StreamSynchronize(BlockingStream) != Success)
    return 1;
  if (StreamSynchronize(NonBlockingStream) != Success)
    return 1;
  if (SyncResult != Success)
    return 1;

  if (StreamDestroy(BlockingStream) != Success)
    return 1;
  if (StreamDestroy(NonBlockingStream) != Success)
    return 1;
  if (FreeHost(BlockingGate) != Success)
    return 1;
  if (FreeHost(NonBlockingGate) != Success)
    return 1;
  if (FreeHost(BlockingOutStorage) != Success)
    return 1;
  if (FreeHost(NonBlockingOutStorage) != Success)
    return 1;
}
