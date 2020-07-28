// RUN: %libomptarget-compile-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-x86_64-pc-linux-gnu
// RUN: %libomptarget-compile-run-and-check-nvptx64-nvidia-cuda

#include <stdio.h>
#include <omp.h>

int main(void) {
  int isHost = -1;

#pragma omp target map(from: isHost)
  { isHost = omp_is_initial_device(); }

  if (isHost < 0) {
    printf("Runtime error, isHost=%d\n", isHost);
  }

  // CHECK: Target region executed on the device
  printf("Target region executed on the %s\n", isHost ? "host" : "device");

  return isHost;
}
