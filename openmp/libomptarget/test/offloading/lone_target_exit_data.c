// Check that a target exit data directive behaves correctly when the runtime
// has not yet been initialized.

// RUN: %libomptarget-compile-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compile-run-and-check-nvptx64-nvidia-cuda

#include <stdio.h>

int main() {
  // CHECK: x = 98
  int x = 98;
  #pragma omp target exit data map(from:x)
  printf("x = %d\n", x);
  return 0;
}
