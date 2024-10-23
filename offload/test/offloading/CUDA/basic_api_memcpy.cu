// RUN: %clang++ -foffload-via-llvm --offload-arch=native %s -o %t
// RUN: %t | %fcheck-generic

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel(int *DevPtr, int N) {
  for (int i = 0; i < N; ++i)
    DevPtr[i]--;
}

int main(int argc, char **argv) {
  int DevNo = 0;
  int Res = 0;
  int *DevPtr;
  auto Err = cudaMalloc(&DevPtr, 42 * sizeof(int));
  if (Err != cudaSuccess)
    return -1;
  int HstPtr[42];
  for (int i = 0; i < 42; ++i) {
    HstPtr[i] = 2;
  }
  Err = cudaMemcpy(DevPtr, HstPtr, 42 * sizeof(int), cudaMemcpyHostToDevice);
  if (Err != cudaSuccess)
    return -1;
  printf("Res: %i\n", Res);
  // CHECK: Res: 0
  kernel<<<1, 1>>>(DevPtr, 42);
  cudaDeviceSynchronize();
  Err = cudaMemcpy(HstPtr, DevPtr, 42 * sizeof(int), cudaMemcpyDeviceToHost);
  if (Err != cudaSuccess)
    return -1;
  for (int i = 0; i < 42; ++i) {
    printf("%i : %i\n", i, HstPtr[i]);
    Res += HstPtr[i];
  }
  printf("Res: %i\n", Res);
  // CHECK: Res: 42
  Err = cudaFree(DevPtr);
  if (Err != cudaSuccess)
    return -1;
}
