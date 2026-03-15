// RUN: %libomptarget-compile-generic && env HSA_XNACK=1 %libomptarget-run-generic 2>&1 | %fcheck-generic

// REQUIRES: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

#define N 1024

int main(int argc, char *argv[]) {
  int fails;
  void *host_alloc = 0, *device_alloc = 0;
  int *a = (int *)malloc(N * sizeof(int));
  int dev = omp_get_default_device();

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
  {
    device_alloc = &a[0];
    a[0] = 99;
  }

  // 'close' is missing, so the runtime must check whether s is actually in
  // shared memory in order to determine whether to transfer data and delete the
  // allocation.
#pragma omp target exit data map(from : a[ : N])

  // CHECK: a has been mapped to the device.
  if (device_alloc != host_alloc)
    printf("a has been mapped to the device.\n");

  // CHECK: a[0]=99
  // CHECK: a is present: 0
  printf("a[0]=%d\n", a[0]);
  printf("a is present: %d\n", omp_target_is_present(a, dev));

  free(a);

  // CHECK: Done!
  printf("Done!\n");

  return 0;
}
