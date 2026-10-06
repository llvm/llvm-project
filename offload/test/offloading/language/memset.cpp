// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.legacy -fgpu-default-stream=legacy -pthread -std=c++17
// RUN: %t.cuda.legacy | %fcheck-generic --check-prefix=LEGACY
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.perthread -fgpu-default-stream=per-thread -pthread -std=c++17
// RUN: %t.cuda.perthread | %fcheck-generic --check-prefix=PERTHREAD
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.legacy -fgpu-default-stream=legacy -pthread -std=c++17
// RUN: %t.hip.legacy | %fcheck-generic --check-prefix=LEGACY
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.perthread -fgpu-default-stream=per-thread -pthread -std=c++17
// RUN: %t.hip.perthread | %fcheck-generic --check-prefix=PERTHREAD
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

__global__ void waitThenSet(volatile int *Gate, unsigned char *Out,
                            unsigned char Value) {
  for (unsigned long long I = 0; I < 1000000000ULL && *Gate == 0; ++I)
    ;
  if (*Gate)
    *Out = Value;
}

int main(int argc, char **argv) {
  unsigned char *Dev = nullptr;
  if (Malloc(&Dev, 4) != Success)
    return 1;

  if (Memset(Dev, 0x2a, 4) != Success)
    return 1;

  unsigned char Host[4] = {};
  if (Memcpy(Host, Dev, sizeof(Host), MemcpyDeviceToHost) != Success)
    return 1;
  printf("memset bytes: %u %u %u %u\n", static_cast<unsigned>(Host[0]),
         static_cast<unsigned>(Host[1]), static_cast<unsigned>(Host[2]),
         static_cast<unsigned>(Host[3]));
  // LEGACY: memset bytes: 42 42 42 42
  // PERTHREAD: memset bytes: 42 42 42 42

  Stream_t BlockingStream = nullptr;
  if (StreamCreateWithFlags(&BlockingStream, StreamDefault) != Success)
    return 1;

  int *Gate = nullptr;
  if (HostAlloc(&Gate, sizeof(int), HostAllocDefault) != Success)
    return 1;
  *Gate = 0;

  if (Memset(Dev, 0, 1) != Success)
    return 1;

  waitThenSet<<<1, 1, 0, BlockingStream>>>(Gate, Dev, 17);

  std::thread Releaser([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    *Gate = 1;
  });

  Error_t MemsetResult = Memset(Dev, 23, 1);

  Releaser.join();
  if (MemsetResult != Success)
    return 1;

  if (StreamSynchronize(BlockingStream) != Success)
    return 1;

  unsigned char Result = 0;
  if (Memcpy(&Result, Dev, 1, MemcpyDeviceToHost) != Success)
    return 1;
  printf("default stream memset result: %u\n", static_cast<unsigned>(Result));
  // LEGACY: default stream memset result: 23
  // PERTHREAD: default stream memset result: 17

  if (StreamDestroy(BlockingStream) != Success)
    return 1;
  if (FreeHost(Gate) != Success)
    return 1;
  if (Free(Dev) != Success)
    return 1;
}
