// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fgpu-default-stream=legacy -pthread -std=c++17
// RUN: %t | %fcheck-generic --check-prefix=CHECK
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fgpu-default-stream=per-thread -pthread -std=c++17
// RUN: %t | %fcheck-generic --check-prefix=CHECK
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

#include <chrono>
#include <cstdio>
#include <thread>

__global__ void waitThenSet(volatile int *Gate, volatile int *Out, int Value) {
  for (unsigned long long I = 0; I < 1000000000ULL && *Gate == 0; ++I)
    ;
  *Out = *Gate ? Value : -Value;
}

int main(int argc, char **argv) {
  hipStream_t BlockingStream = nullptr;
  if (hipStreamCreateWithFlags(&BlockingStream, hipStreamDefault) != hipSuccess)
    return 1;
  hipStream_t NonBlockingStream = nullptr;
  if (hipStreamCreateWithFlags(&NonBlockingStream, hipStreamNonBlocking) !=
      hipSuccess)
    return 1;

  int *BlockingGate = nullptr;
  int *NonBlockingGate = nullptr;
  int *BlockingOutStorage = nullptr;
  int *NonBlockingOutStorage = nullptr;
  if (hipHostAlloc(&BlockingGate, sizeof(int), hipHostAllocDefault) !=
      hipSuccess)
    return 1;
  if (hipHostAlloc(&NonBlockingGate, sizeof(int), hipHostAllocDefault) !=
      hipSuccess)
    return 1;
  if (hipHostAlloc(&BlockingOutStorage, sizeof(int), hipHostAllocDefault) !=
      hipSuccess)
    return 1;
  if (hipHostAlloc(&NonBlockingOutStorage, sizeof(int), hipHostAllocDefault) !=
      hipSuccess)
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

  hipError_t SyncResult = hipDeviceSynchronize();

  if (SyncResult == hipSuccess) {
    printf("device sync waited on blocking stream: %d\n", *BlockingOut);
    // CHECK: device sync waited on blocking stream: 17
    printf("device sync waited on nonblocking stream: %d\n", *NonBlockingOut);
    // CHECK: device sync waited on nonblocking stream: 23
  }

  Releaser.join();
  if (hipStreamSynchronize(BlockingStream) != hipSuccess)
    return 1;
  if (hipStreamSynchronize(NonBlockingStream) != hipSuccess)
    return 1;
  if (SyncResult != hipSuccess)
    return 1;

  if (hipStreamDestroy(BlockingStream) != hipSuccess)
    return 1;
  if (hipStreamDestroy(NonBlockingStream) != hipSuccess)
    return 1;
  if (hipFreeHost(BlockingGate) != hipSuccess)
    return 1;
  if (hipFreeHost(NonBlockingGate) != hipSuccess)
    return 1;
  if (hipFreeHost(BlockingOutStorage) != hipSuccess)
    return 1;
  if (hipFreeHost(NonBlockingOutStorage) != hipSuccess)
    return 1;
}
