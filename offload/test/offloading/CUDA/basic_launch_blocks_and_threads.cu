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

__global__ void square(int *A) {
  __scoped_atomic_fetch_add(A, 1, __ATOMIC_SEQ_CST, __MEMORY_SCOPE_DEVICE);
}

int main(int argc, char **argv) {
  int DevNo = 0;
  int *Ptr = reinterpret_cast<int *>(llvm_omp_target_alloc_shared(4, DevNo));
  *Ptr = 0;
  printf("Ptr %p, *Ptr: %i\n", Ptr, *Ptr);
  // CHECK: Ptr [[Ptr:0x.*]], *Ptr: 0
  square<<<7, 6>>>(Ptr);
  printf("Ptr %p, *Ptr: %i\n", Ptr, *Ptr);
  // CHECK: Ptr [[Ptr]], *Ptr: 42
  llvm_omp_target_free_shared(Ptr, DevNo);
}
