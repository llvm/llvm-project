// RUN: %libomptarget-compile-run-and-check-generic

// REQUIRES: unified_shared_memory
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// Fails on amdgcn with error: GPU Memory Error
// XFAIL: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

#define N 1024

int main(int argc, char *argv[]) {
  int fails;
  void *host_alloc = 0, *device_alloc = 0;
  int *a = (int *)malloc(N * sizeof(int));

  // Init
  for (int i = 0; i < N; ++i) {
    a[i] = 10;
  }
  host_alloc = &a[0];

  //
  // map + target no close
  //
#pragma omp target data map(tofrom : a[ : N]) map(tofrom : device_alloc)
  {
#pragma omp target map(tofrom : device_alloc)
    { device_alloc = &a[0]; }
  }

  // CHECK: a used from unified memory.
  if (device_alloc == host_alloc)
    printf("a used from unified memory.\n");

  //
  // map + target with close
  //
  device_alloc = 0;
#pragma omp target data map(close, tofrom : a[ : N]) map(tofrom : device_alloc)
  {
#pragma omp target map(tofrom : device_alloc)
    { device_alloc = &a[0]; }
  }
  // CHECK: a copied to device.
  if (device_alloc != host_alloc)
    printf("a copied to device.\n");

  //
  // map + use_device_ptr no close
  //
  device_alloc = 0;
#pragma omp target data map(tofrom : a[ : N]) use_device_ptr(a)
  { device_alloc = &a[0]; }

  // CHECK: a used from unified memory with use_device_ptr.
  if (device_alloc == host_alloc)
    printf("a used from unified memory with use_device_ptr.\n");

  //
  // map + use_device_ptr close
  //
  device_alloc = 0;
#pragma omp target data map(close, tofrom : a[ : N]) use_device_ptr(a)
  { device_alloc = &a[0]; }

  // CHECK: a used from device memory with use_device_ptr.
  if (device_alloc != host_alloc)
    printf("a used from device memory with use_device_ptr.\n");

  //
  // map enter/exit + close
  //
  device_alloc = 0;
#pragma omp target enter data map(close, to : a[ : N])

#pragma omp target map(from : device_alloc)
  { device_alloc = &a[0]; }

#pragma omp target exit data map(from : a[ : N])

  // CHECK: a has been mapped to the device.
  if (device_alloc != host_alloc)
    printf("a has been mapped to the device.\n");

  free(a);

  // CHECK: Done!
  printf("Done!\n");

  return 0;
}
