// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fgpu-default-stream=legacy -pthread -std=c++17
// RUN: %t | %fcheck-generic --check-prefix=LEGACY
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fgpu-default-stream=per-thread -pthread -std=c++17
// RUN: %t | %fcheck-generic --check-prefix=PERTHREAD
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

__global__ void waitThenSet(volatile int *Gate, unsigned char *Out,
                            unsigned char Value) {
  for (unsigned long long I = 0; I < 1000000000ULL && *Gate == 0; ++I)
    ;
  if (*Gate)
    *Out = Value;
}

int main(int argc, char **argv) {
  unsigned char *Dev = nullptr;
  if (hipMalloc(&Dev, 4) != hipSuccess)
    return 1;

  if (hipMemset(Dev, 0x2a, 4) != hipSuccess)
    return 1;

  unsigned char Host[4] = {};
  if (hipMemcpy(Host, Dev, sizeof(Host), hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  printf("memset bytes: %u %u %u %u\n", static_cast<unsigned>(Host[0]),
         static_cast<unsigned>(Host[1]), static_cast<unsigned>(Host[2]),
         static_cast<unsigned>(Host[3]));
  // LEGACY: memset bytes: 42 42 42 42
  // PERTHREAD: memset bytes: 42 42 42 42

  hipStream_t BlockingStream = nullptr;
  if (hipStreamCreateWithFlags(&BlockingStream, hipStreamDefault) != hipSuccess)
    return 1;

  int *Gate = nullptr;
  if (hipHostAlloc(&Gate, sizeof(int), hipHostAllocDefault) != hipSuccess)
    return 1;
  *Gate = 0;

  if (hipMemset(Dev, 0, 1) != hipSuccess)
    return 1;

  waitThenSet<<<1, 1, 0, BlockingStream>>>(Gate, Dev, 17);

  std::thread Releaser([&]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    *Gate = 1;
  });

  hipError_t MemsetResult = hipMemset(Dev, 23, 1);

  Releaser.join();
  if (MemsetResult != hipSuccess)
    return 1;

  if (hipStreamSynchronize(BlockingStream) != hipSuccess)
    return 1;

  unsigned char Result = 0;
  if (hipMemcpy(&Result, Dev, 1, hipMemcpyDeviceToHost) != hipSuccess)
    return 1;
  printf("default stream memset result: %u\n", static_cast<unsigned>(Result));
  // LEGACY: default stream memset result: 23
  // PERTHREAD: default stream memset result: 17

  if (hipStreamDestroy(BlockingStream) != hipSuccess)
    return 1;
  if (hipFreeHost(Gate) != hipSuccess)
    return 1;
  if (hipFree(Dev) != hipSuccess)
    return 1;
}
