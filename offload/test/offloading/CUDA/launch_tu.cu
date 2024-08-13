// clang-format off
// RUN: %clang++ -foffload-via-llvm --offload-arch=native %s -o %t.launch_tu.o -c
// RUN: %clang++ -foffload-via-llvm --offload-arch=native -x cuda %S/kernel_tu.cu.inc -o %t.kernel_tu.o -c
// RUN: %clang++ -foffload-via-llvm --offload-arch=native %t.launch_tu.o %t.kernel_tu.o -o %t
// RUN: %t | %fcheck-generic
// clang-format on

// UNSUPPORTED: aarch64-unknown-linux-gnu
// UNSUPPORTED: aarch64-unknown-linux-gnu-LTO
// UNSUPPORTED: x86_64-pc-linux-gnu
// UNSUPPORTED: x86_64-pc-linux-gnu-LTO

#include <stdio.h>

extern "C" {
void *llvm_omp_target_alloc_shared(size_t Size, int DeviceNum);
void llvm_omp_target_free_shared(void *DevicePtr, int DeviceNum);
}

extern __global__ void square(int *A);

int main(int argc, char **argv) {
  int DevNo = 0;
  int *Ptr = reinterpret_cast<int *>(llvm_omp_target_alloc_shared(4, DevNo));
  *Ptr = 7;
  printf("Ptr %p, *Ptr: %i\n", Ptr, *Ptr);
  // CHECK: Ptr [[Ptr:0x.*]], *Ptr: 7
  square<<<1, 1>>>(Ptr);
  printf("Ptr %p, *Ptr: %i\n", Ptr, *Ptr);
  // CHECK: Ptr [[Ptr]], *Ptr: 42
  llvm_omp_target_free_shared(Ptr, DevNo);
}
