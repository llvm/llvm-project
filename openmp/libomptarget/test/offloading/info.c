// RUN: %libomptarget-compile-nvptx64-nvidia-cuda && env LIBOMPTARGET_INFO=1 %libomptarget-run-nvptx64-nvidia-cuda 2>&1 | %fcheck-nvptx64-nvidia-cuda -allow-empty -check-prefix=INFO

#include <stdio.h>
#include <omp.h>

int main() {
    int ptr = 1;

// INFO: CUDA device {{[0-9]+}} info: Device supports up to {{[0-9]+}} CUDA blocks and {{[0-9]+}} threads with a warp size of {{[0-9]+}}
// INFO: CUDA device {{[0-9]+}} info: Launching kernel {{.*}} with {{[0-9]+}} blocks and {{[0-9]+}} threads in Generic mode
#pragma omp target map(tofrom:ptr)
  {ptr = 1;}

  return 0;
}
