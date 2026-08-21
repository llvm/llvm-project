// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -pthread -std=c++17
// RUN: %t | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fopenmp -pthread -std=c++17
// RUN: %t | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

#include <cstdio>
#include <mutex>
#include <thread>

static std::mutex PrintMutex;

static void printError(int ThreadId, const char *Label, hipError_t Error) {
  std::lock_guard<std::mutex> Lock(PrintMutex);
  printf("thread %d %s: %s\n", ThreadId, Label, hipGetErrorName(Error));
  std::fflush(stdout);
}

__global__ void errorKernel(float *d_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_out[idx] = idx * 0.5f;
}

void runTask(int ThreadId) {
  const int N = 1 << 20;
  size_t Bytes = N * sizeof(float);

  float *d_data;
  hipMalloc(&d_data, Bytes);
  printError(ThreadId, "hipMalloc", hipGetLastError());

  ThreadId == 1 ? errorKernel<<<4096, 256>>>(d_data)
                : errorKernel<<<4096, 0>>>(d_data);
  printError(ThreadId, "kernel launch", hipPeekAtLastError());

  printError(ThreadId, "kernel launch get", hipGetLastError());

  printError(ThreadId, "kernel launch get again", hipGetLastError());

  hipDeviceSynchronize();
  hipFree(d_data);
}

int main() {
  printError(0, "initial", hipPeekAtLastError());
  // CHECK: thread 0 initial: hipSuccess

  std::thread t1(runTask, 1);
  std::thread t2(runTask, 2);

  t1.join();
  t2.join();
  // CHECK-DAG: thread 1 hipMalloc: hipSuccess
  // CHECK-DAG: thread 2 hipMalloc: hipSuccess
  // CHECK-DAG: thread 1 kernel launch: hipSuccess
  // CHECK-DAG: thread 2 kernel launch: hipErrorInvalidConfiguration
  // CHECK-DAG: thread 1 kernel launch get: hipSuccess
  // CHECK-DAG: thread 2 kernel launch get: hipErrorInvalidConfiguration
  // CHECK-DAG: thread 1 kernel launch get again: hipSuccess
  // CHECK-DAG: thread 2 kernel launch get again: hipSuccess

  std::thread t3(runTask, 3);
  t3.join();
  // CHECK: thread 3 hipMalloc: hipSuccess
  // CHECK: thread 3 kernel launch: hipErrorInvalidConfiguration
  // CHECK: thread 3 kernel launch get: hipErrorInvalidConfiguration
  // CHECK: thread 3 kernel launch get again: hipSuccess

  printError(0, "joined", hipGetLastError());
  // CHECK: thread 0 joined: hipSuccess

  return 0;
}
