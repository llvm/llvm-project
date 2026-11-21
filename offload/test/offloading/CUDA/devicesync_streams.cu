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
  cudaStream_t BlockingStream = nullptr;
  if (cudaStreamCreateWithFlags(&BlockingStream, cudaStreamDefault) !=
      cudaSuccess)
    return 1;
  cudaStream_t NonBlockingStream = nullptr;
  if (cudaStreamCreateWithFlags(&NonBlockingStream, cudaStreamNonBlocking) !=
      cudaSuccess)
    return 1;

  int *BlockingGate = nullptr;
  int *NonBlockingGate = nullptr;
  int *BlockingOutStorage = nullptr;
  int *NonBlockingOutStorage = nullptr;
  if (cudaHostAlloc(&BlockingGate, sizeof(int), cudaHostAllocDefault) !=
      cudaSuccess)
    return 1;
  if (cudaHostAlloc(&NonBlockingGate, sizeof(int), cudaHostAllocDefault) !=
      cudaSuccess)
    return 1;
  if (cudaHostAlloc(&BlockingOutStorage, sizeof(int), cudaHostAllocDefault) !=
      cudaSuccess)
    return 1;
  if (cudaHostAlloc(&NonBlockingOutStorage, sizeof(int),
                    cudaHostAllocDefault) != cudaSuccess)
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

  cudaError_t SyncResult = cudaDeviceSynchronize();

  if (SyncResult == cudaSuccess) {
    printf("device sync waited on blocking stream: %d\n", *BlockingOut);
    // CHECK: device sync waited on blocking stream: 17
    printf("device sync waited on nonblocking stream: %d\n", *NonBlockingOut);
    // CHECK: device sync waited on nonblocking stream: 23
  }

  Releaser.join();
  if (cudaStreamSynchronize(BlockingStream) != cudaSuccess)
    return 1;
  if (cudaStreamSynchronize(NonBlockingStream) != cudaSuccess)
    return 1;
  if (SyncResult != cudaSuccess)
    return 1;

  if (cudaStreamDestroy(BlockingStream) != cudaSuccess)
    return 1;
  if (cudaStreamDestroy(NonBlockingStream) != cudaSuccess)
    return 1;
  if (cudaFreeHost(BlockingGate) != cudaSuccess)
    return 1;
  if (cudaFreeHost(NonBlockingGate) != cudaSuccess)
    return 1;
  if (cudaFreeHost(BlockingOutStorage) != cudaSuccess)
    return 1;
  if (cudaFreeHost(NonBlockingOutStorage) != cudaSuccess)
    return 1;
}
