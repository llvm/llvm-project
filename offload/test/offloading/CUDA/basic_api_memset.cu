// RUN: %clang++ -foffload-via-llvm --offload-arch=native %s -o %t
// RUN: %t | %fcheck-generic

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
void *llvm_omp_target_alloc_shared(size_t Size, int DeviceNum);
void llvm_omp_target_free_shared(void *DevicePtr, int DeviceNum);
}

__global__ void kernel(int *A, int *DevPtr, int N) {
  for (int i = 0; i < N; ++i)
    *A += DevPtr[i];
  *A *= -1;
}

int main(int argc, char **argv) {
  int DevNo = 0;
  int *Ptr = reinterpret_cast<int *>(llvm_omp_target_alloc_shared(4, DevNo));
  int *DevPtr;
  auto Err = cudaMalloc(&DevPtr, 42 * sizeof(int));
  if (Err != cudaSuccess)
    return -1;
  Err = cudaMemset(DevPtr, -1, 42 * sizeof(int));
  if (Err != cudaSuccess)
    return -1;
  *Ptr = 0;
  printf("Ptr %p, *Ptr: %i\n", Ptr, *Ptr);
  // CHECK: Ptr [[Ptr:0x.*]], *Ptr: 0
  kernel<<<1, 1>>>(Ptr, DevPtr, 42);
  cudaDeviceSynchronize();
  printf("Ptr %p, *Ptr: %i\n", Ptr, *Ptr);
  // CHECK: Ptr [[Ptr]], *Ptr: 42
  Err = cudaFree(DevPtr);
  if (Err != cudaSuccess)
    return -1;
  llvm_omp_target_free_shared(Ptr, DevNo);
}
