// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// amdgpu runtime crash
// Fails on nvptx with error: an illegal memory access was encountered
// UNSUPPORTED: amdgcn-amd-amdhsa
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// UNSUPPORTED: intelgpu

// This is an implementation-specific test (it does not reflect official OpenMP
// expectations). The test is to ensure that we do the optimization to
// "allocate" a new device copy for maps with `close` modifier, but only when a
// matching entry doesn't already exist on the device.

#include <assert.h>

int x, y;
#pragma omp requires unified_shared_memory

int main() {
  int *xaddr_device, *yaddr_device;
#pragma omp target data map(tofrom : x)
#pragma omp target map(close, tofrom : x, y)                                   \
    map(from : xaddr_device, yaddr_device)
  {
    xaddr_device = &x;
    yaddr_device = &y;
  }

  assert(xaddr_device == &x && "Mapped variable should not allocate on close.");
  assert(yaddr_device != &y && "Unmapped variable should allocate on close.");
}
