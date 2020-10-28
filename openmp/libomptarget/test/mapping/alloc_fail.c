// RUN: %libomptarget-compile-aarch64-unknown-linux-gnu
// RUN: %libomptarget-run-fail-aarch64-unknown-linux-gnu 2>&1 \
// RUN: | %fcheck-aarch64-unknown-linux-gnu

// RUN: %libomptarget-compile-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-run-fail-powerpc64-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64-ibm-linux-gnu

// RUN: %libomptarget-compile-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-run-fail-powerpc64le-ibm-linux-gnu 2>&1 \
// RUN: | %fcheck-powerpc64le-ibm-linux-gnu

// RUN: %libomptarget-compile-x86_64-pc-linux-gnu
// RUN: %libomptarget-run-fail-x86_64-pc-linux-gnu 2>&1 \
// RUN: | %fcheck-x86_64-pc-linux-gnu

// RUN: %libomptarget-compile-nvptx64-nvidia-cuda
// RUN: %libomptarget-run-fail-nvptx64-nvidia-cuda 2>&1 \
// RUN: | %fcheck-nvptx64-nvidia-cuda

// CHECK: Libomptarget message: explicit extension not allowed: host address specified is 0x{{.*}} (8 bytes), but device allocation maps to host at 0x{{.*}} (8 bytes)
// CHECK: Libomptarget error: Call to getOrAllocTgtPtr returned null pointer (device failure or illegal mapping).
// CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory

int main() {
  int arr[4] = {0, 1, 2, 3};
#pragma omp target data map(alloc: arr[0:2])
#pragma omp target data map(alloc: arr[1:2])
  ;
  return 0;
}
