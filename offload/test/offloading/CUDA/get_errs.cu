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
#include <cuda_runtime.h>
#include <mutex>
#include <thread>

static std::mutex PrintMutex;

static void printError(int ThreadId, const char *Label, cudaError_t Error) {
  std::lock_guard<std::mutex> Lock(PrintMutex);
  printf("thread %d %s: %s\n", ThreadId, Label, cudaGetErrorName(Error));
  std::fflush(stdout);
}

__global__ void errorKernel(float *d_out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  d_out[idx] = idx * 0.5f;
}

void runTask(int ThreadId) {
  const int N = 1 << 20;
  size_t bytes = N * sizeof(float);

  float *d_data;
  cudaMalloc(&d_data, bytes);
  printError(ThreadId, "cudaMalloc", cudaGetLastError());

  ThreadId == 1 ? errorKernel<<<4096, 256>>>(d_data)
                : errorKernel<<<4096, 0>>>(d_data);
  printError(ThreadId, "kernel launch", cudaPeekAtLastError());

  printError(ThreadId, "kernel launch get", cudaGetLastError());

  printError(ThreadId, "kernel launch get again", cudaGetLastError());

  cudaDeviceSynchronize();
  cudaFree(d_data);
}

int main() {
  printError(0, "initial", cudaPeekAtLastError());
  // CHECK: thread 0 initial: cudaSuccess

  std::thread t1(runTask, 1);
  std::thread t2(runTask, 2);

  t1.join();
  t2.join();
  // CHECK-DAG: thread 1 cudaMalloc: cudaSuccess
  // CHECK-DAG: thread 2 cudaMalloc: cudaSuccess
  // CHECK-DAG: thread 1 kernel launch: cudaSuccess
  // CHECK-DAG: thread 2 kernel launch: cudaErrorInvalidConfiguration
  // CHECK-DAG: thread 1 kernel launch get: cudaSuccess
  // CHECK-DAG: thread 2 kernel launch get: cudaErrorInvalidConfiguration
  // CHECK-DAG: thread 1 kernel launch get again: cudaSuccess
  // CHECK-DAG: thread 2 kernel launch get again: cudaSuccess

  std::thread t3(runTask, 3);
  t3.join();
  // CHECK: thread 3 cudaMalloc: cudaSuccess
  // CHECK: thread 3 kernel launch: cudaErrorInvalidConfiguration
  // CHECK: thread 3 kernel launch get: cudaErrorInvalidConfiguration
  // CHECK: thread 3 kernel launch get again: cudaSuccess

  printError(0, "joined", cudaGetLastError());
  // CHECK: thread 0 joined: cudaSuccess

  return 0;
}
