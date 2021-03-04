// Check that mandatory offloading causes various offloading directives to fail
// when omp_get_num_devices() == 0 even if the requested device is the initial
// device.  This behavior is proposed for OpenMP 5.2 in OpenMP spec github
// issue 2669.

// RUN: %libomptarget-compile-nvptx64-nvidia-cuda -DDIR=target
// RUN: env OMP_TARGET_OFFLOAD=mandatory CUDA_VISIBLE_DEVICES= \
// RUN:   %libomptarget-run-fail-nvptx64-nvidia-cuda 2>&1 | \
// RUN:   %fcheck-nvptx64-nvidia-cuda

// RUN: %libomptarget-compile-nvptx64-nvidia-cuda -DDIR='target teams'
// RUN: env OMP_TARGET_OFFLOAD=mandatory CUDA_VISIBLE_DEVICES= \
// RUN:   %libomptarget-run-fail-nvptx64-nvidia-cuda 2>&1 | \
// RUN:   %fcheck-nvptx64-nvidia-cuda

// RUN: %libomptarget-compile-nvptx64-nvidia-cuda -DDIR='target data map(X)'
// RUN: env OMP_TARGET_OFFLOAD=mandatory CUDA_VISIBLE_DEVICES= \
// RUN:   %libomptarget-run-fail-nvptx64-nvidia-cuda 2>&1 | \
// RUN:   %fcheck-nvptx64-nvidia-cuda

// RUN: %libomptarget-compile-nvptx64-nvidia-cuda \
// RUN:   -DDIR='target enter data map(to:X)'
// RUN: env OMP_TARGET_OFFLOAD=mandatory CUDA_VISIBLE_DEVICES= \
// RUN:   %libomptarget-run-fail-nvptx64-nvidia-cuda 2>&1 | \
// RUN:   %fcheck-nvptx64-nvidia-cuda

// RUN: %libomptarget-compile-nvptx64-nvidia-cuda \
// RUN:   -DDIR='target exit data map(from:X)'
// RUN: env OMP_TARGET_OFFLOAD=mandatory CUDA_VISIBLE_DEVICES= \
// RUN:   %libomptarget-run-fail-nvptx64-nvidia-cuda 2>&1 | \
// RUN:   %fcheck-nvptx64-nvidia-cuda

// RUN: %libomptarget-compile-nvptx64-nvidia-cuda \
// RUN:   -DDIR='target update to(X)'
// RUN: env OMP_TARGET_OFFLOAD=mandatory CUDA_VISIBLE_DEVICES= \
// RUN:   %libomptarget-run-fail-nvptx64-nvidia-cuda 2>&1 | \
// RUN:   %fcheck-nvptx64-nvidia-cuda

// RUN: %libomptarget-compile-nvptx64-nvidia-cuda \
// RUN:   -DDIR='target update from(X)'
// RUN: env OMP_TARGET_OFFLOAD=mandatory CUDA_VISIBLE_DEVICES= \
// RUN:   %libomptarget-run-fail-nvptx64-nvidia-cuda 2>&1 | \
// RUN:   %fcheck-nvptx64-nvidia-cuda

#include <omp.h>
#include <stdio.h>

// CHECK: Libomptarget fatal error 1: failure of target construct while offloading is mandatory
int main(void) {
  int X;
  #pragma omp DIR device(omp_get_initial_device())
  ;
  return 0;
}
