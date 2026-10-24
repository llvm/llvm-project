// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda -pthread -std=c++17
// RUN: %t.cuda | %fcheck-generic --check-prefix=CUDA
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x cuda -DOFFLOAD_TEST_LANGUAGE=cuda %s -o %t.cuda.omp -fopenmp -pthread -std=c++17
// RUN: %t.cuda.omp | %fcheck-generic --check-prefix=CUDA
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip -pthread -std=c++17
// RUN: %t.hip | %fcheck-generic --check-prefix=HIP
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native -x hip -DOFFLOAD_TEST_LANGUAGE=hip %s -o %t.hip.omp -fopenmp -pthread -std=c++17
// RUN: %t.hip.omp | %fcheck-generic --check-prefix=HIP
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: amdgcn-amd-amdhsa-LTO
// UNSUPPORTED: amdgpu-amd-amdhsa-LTO
// UNSUPPORTED: intelgpu

// clang-format off
#include <cstdio>
#include <mutex>
#include <thread>
#include "Inputs/DefineTestLanguageNames.inc"
// clang-format on

static std::mutex PrintMutex;

static void printError(int ThreadId, const char *Label, Error_t Error) {
  std::lock_guard<std::mutex> Lock(PrintMutex);
  printf("thread %d %s: %s\n", ThreadId, Label, GetErrorName(Error));
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
  Malloc(&d_data, bytes);
  printError(ThreadId, "Malloc", GetLastError());

  ThreadId == 1 ? errorKernel<<<4096, 256>>>(d_data)
                : errorKernel<<<4096, 0>>>(d_data);
  printError(ThreadId, "kernel launch", PeekAtLastError());

  printError(ThreadId, "kernel launch get", GetLastError());

  printError(ThreadId, "kernel launch get again", GetLastError());

  DeviceSynchronize();
  Free(d_data);
}

int main() {
  printError(0, "initial", PeekAtLastError());
  // CUDA: thread 0 initial: cudaSuccess
  // HIP: thread 0 initial: hipSuccess

  std::thread t1(runTask, 1);
  std::thread t2(runTask, 2);

  t1.join();
  t2.join();
  // CUDA-DAG: thread 1 Malloc: cudaSuccess
  // HIP-DAG: thread 1 Malloc: hipSuccess
  // CUDA-DAG: thread 2 Malloc: cudaSuccess
  // HIP-DAG: thread 2 Malloc: hipSuccess
  // CUDA-DAG: thread 1 kernel launch: cudaSuccess
  // HIP-DAG: thread 1 kernel launch: hipSuccess
  // CUDA-DAG: thread 2 kernel launch: cudaErrorInvalidConfiguration
  // HIP-DAG: thread 2 kernel launch: hipErrorInvalidConfiguration
  // CUDA-DAG: thread 1 kernel launch get: cudaSuccess
  // HIP-DAG: thread 1 kernel launch get: hipSuccess
  // CUDA-DAG: thread 2 kernel launch get: cudaErrorInvalidConfiguration
  // HIP-DAG: thread 2 kernel launch get: hipErrorInvalidConfiguration
  // CUDA-DAG: thread 1 kernel launch get again: cudaSuccess
  // HIP-DAG: thread 1 kernel launch get again: hipSuccess
  // CUDA-DAG: thread 2 kernel launch get again: cudaSuccess
  // HIP-DAG: thread 2 kernel launch get again: hipSuccess

  std::thread t3(runTask, 3);
  t3.join();
  // CUDA: thread 3 Malloc: cudaSuccess
  // HIP: thread 3 Malloc: hipSuccess
  // CUDA: thread 3 kernel launch: cudaErrorInvalidConfiguration
  // HIP: thread 3 kernel launch: hipErrorInvalidConfiguration
  // CUDA: thread 3 kernel launch get: cudaErrorInvalidConfiguration
  // HIP: thread 3 kernel launch get: hipErrorInvalidConfiguration
  // CUDA: thread 3 kernel launch get again: cudaSuccess
  // HIP: thread 3 kernel launch get again: hipSuccess

  printError(0, "joined", GetLastError());
  // CUDA: thread 0 joined: cudaSuccess
  // HIP: thread 0 joined: hipSuccess

  return 0;
}
