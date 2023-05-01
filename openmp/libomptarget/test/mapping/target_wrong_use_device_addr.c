// RUN: %libomptarget-compile-generic -fopenmp-version=51 -g
// RUN: env LIBOMPTARGET_INFO=64 %libomptarget-run-fail-generic 2>&1 \
// RUN: | %fcheck-generic

#include <stdio.h>

int main() {
  float arr[10];
  float *x = &arr[0];

  // CHECK: host addr=0x[[#%x,HOST_ADDR:]]
  fprintf(stderr, "host addr=%p\n", x);

#pragma omp target data map(to : x [0:10])
  {
// CHECK: Libomptarget device 0 info: variable x does not have a valid device
// counterpart
#pragma omp target data use_device_addr(x)
    {
      // CHECK-NOT: device addr=0x[[#%x,HOST_ADDR:]]
      fprintf(stderr, "device addr=%p\n", x);
    }
  }

  return 0;
}

