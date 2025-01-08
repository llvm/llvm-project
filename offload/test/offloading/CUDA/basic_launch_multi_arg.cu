// clang-format off
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t
// RUN: %t | %fcheck-generic
// RUN: %clang++ %flags -foffload-via-llvm --offload-arch=native %s -o %t -fopenmp 
// RUN: %t | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-unknown-linux-gnu
// UNSUPPORTED: x86_64-unknown-linux-gnu-LTO

#include <stdio.h>

extern "C" {
void *llvm_omp_target_alloc_shared(size_t Size, int DeviceNum);
void llvm_omp_target_free_shared(void *DevicePtr, int DeviceNum);
}

__global__ void square(int *Dst, short Q, int *Src, short P) {
  *Dst = (Src[0] + Src[1]) * (Q + P);
  Src[0] = Q;
  Src[1] = P;
}

int main(int argc, char **argv) {
  int DevNo = 0;
  int *Ptr = reinterpret_cast<int *>(llvm_omp_target_alloc_shared(4, DevNo));
  int *Src = reinterpret_cast<int *>(llvm_omp_target_alloc_shared(8, DevNo));
  *Ptr = 7;
  Src[0] = -2;
  Src[1] = 8;
  printf("Ptr %p, *Ptr: %i\n", Ptr, *Ptr);
  // CHECK: Ptr [[Ptr:0x.*]], *Ptr: 7
  printf("Src: %i : %i\n", Src[0], Src[1]);
  // CHECK: Src: -2 : 8
  square<<<1, 1>>>(Ptr, 3, Src, 4);
  printf("Ptr %p, *Ptr: %i\n", Ptr, *Ptr);
  // CHECK: Ptr [[Ptr]], *Ptr: 42
  printf("Src: %i : %i\n", Src[0], Src[1]);
  // CHECK: Src: 3 : 4
  llvm_omp_target_free_shared(Ptr, DevNo);
}
