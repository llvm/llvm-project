// clang-format off
// RUN: %libomptarget-compilexx-generic && env HSA_XNACK=1 LIBOMPTARGET_INFO=-1 %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

// REQUIRES: amdgcn-amd-amdhsa

#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

#define N 1024

int main(int argc, char *argv[]) {
  void *host_alloc = nullptr, *device_alloc = nullptr;
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

// clang-format off
// CHECK: Entering OpenMP data region with being_mapper at {{.*}} with 2 arguments:
// CHECK: Creating new map entry ONLY with HstPtrBase=[[A_HST_PTR:0x.*]], HstPtrBegin=[[A_HST_PTR]], TgtAllocBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_HST_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
// CHECK: Creating new map entry ONLY with HstPtrBase=[[DEVICE_ALLOC_HST_PTR:0x.*]], HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtAllocBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1, HoldRefCount=0
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[A_HST_PTR]]
// CHECK: [[DEVICE_ALLOC_HST_PTR]]
// clang-format on
#pragma omp target data map(tofrom : a[ : N]) map(tofrom : device_alloc)
  {
// clang-format off
// CHECK: Entering OpenMP kernel at {{.*}} with 2 arguments:
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=2 (incremented), HoldRefCount=0
// CHECK: Mapping exists (implicit) with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_HST_PTR]], Size=0, DynRefCount=2 (incremented), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=2 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_HST_PTR]], Size=0, DynRefCount=2 (update suppressed), HoldRefCount=0
// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode
// clang-format on
#pragma omp target map(tofrom : device_alloc)
    { device_alloc = &a[0]; }
  }
  // clang-format off
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_HST_PTR]], Size=0, DynRefCount=1 (decremented), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1 (decremented), HoldRefCount=0
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[A_HST_PTR]]
// CHECK: [[DEVICE_ALLOC_HST_PTR]]
// CHECK: Exiting OpenMP data region with end_mapper at {{.*}} with 2 arguments:
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_HST_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Removing map entry with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8
// CHECK: Removing map entry with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_HST_PTR]], Size=4096
  // clang-format on
  if (device_alloc == host_alloc)
    printf("a used from unified memory.\n");

  //
  // map + target with close
  //
  // clang-format off
// CHECK: Entering OpenMP data region with being_mapper at {{.*}} with 2 arguments:
// CHECK: Creating new map entry with HstPtrBase=[[A_HST_PTR]], HstPtrBegin=[[A_HST_PTR]], TgtAllocBegin=[[A_DEV_PTR:0x.*]], TgtPtrBegin=[[A_DEV_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
// CHECK: Copying data from host to device, HstPtr=[[A_HST_PTR]], TgtPtr=[[A_DEV_PTR]], Size=4096
// CHECK: Creating new map entry ONLY with HstPtrBase=[[DEVICE_ALLOC_HST_PTR]], HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtAllocBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1, HoldRefCount=0
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[A_HST_PTR]]
// CHECK: [[DEVICE_ALLOC_HST_PTR]]
  // clang-format on
  device_alloc = 0;
#pragma omp target data map(close, tofrom : a[ : N]) map(tofrom : device_alloc)
  {
// clang-format off
// CHECK: Entering OpenMP kernel at {{.*}} with 2 arguments:
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=2 (incremented), HoldRefCount=0
// CHECK: Mapping exists (implicit) with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=0, DynRefCount=2 (incremented), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=2 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=0, DynRefCount=2 (update suppressed), HoldRefCount=0
// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode
// clang-format on
#pragma omp target map(tofrom : device_alloc)
    { device_alloc = &a[0]; }
  }
  // clang-format off
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=0, DynRefCount=1 (decremented), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1 (decremented), HoldRefCount=0
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[A_HST_PTR]]
// CHECK: [[DEVICE_ALLOC_HST_PTR]]
// CHECK: Exiting OpenMP data region with end_mapper at {{.*}} with 2 arguments:
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Copying data from device to host, TgtPtr=[[A_DEV_PTR]], HstPtr=[[A_HST_PTR]], Size=4096
// CHECK: Removing map entry with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8
// CHECK: Removing map entry with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=4096
  // clang-format on
  if (device_alloc != host_alloc)
    printf("a copied to device.\n");

  //
  // map + use_device_ptr no close
  //
  // clang-format off
// CHECK: Entering OpenMP data region with being_mapper at {{.*}} with 1 arguments:
// CHECK: Creating new map entry ONLY with HstPtrBase=[[A_HST_PTR]], HstPtrBegin=[[A_HST_PTR]], TgtAllocBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_HST_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[A_HST_PTR]]
  // clang-format on
  device_alloc = 0;
#pragma omp target data map(tofrom : a[ : N]) use_device_ptr(a)
  { device_alloc = &a[0]; }
  // clang-format off
// CHECK: Exiting OpenMP data region with end_mapper at {{.*}} with 1 arguments:
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_HST_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Removing map entry with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_HST_PTR]], Size=4096
  // clang-format on
  if (device_alloc == host_alloc)
    printf("a used from unified memory with use_device_ptr.\n");

  //
  // map + use_device_ptr close
  //
  // clang-format off
// CHECK: Entering OpenMP data region with being_mapper at {{.*}} with 1 arguments:
// CHECK: Creating new map entry with HstPtrBase=[[A_HST_PTR]], HstPtrBegin=[[A_HST_PTR]], TgtAllocBegin=[[A_DEV_PTR:0x.*]], TgtPtrBegin=[[A_DEV_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
// CHECK: Copying data from host to device, HstPtr=[[A_HST_PTR]], TgtPtr=[[A_DEV_PTR]], Size=4096
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[A_HST_PTR]]
  // clang-format on
  device_alloc = 0;
#pragma omp target data map(close, tofrom : a[ : N]) use_device_ptr(a)
  { device_alloc = &a[0]; }
  // clang-format off
// CHECK: Exiting OpenMP data region with end_mapper at {{.*}} with 1 arguments:
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Copying data from device to host, TgtPtr=[[A_DEV_PTR]], HstPtr=[[A_HST_PTR]], Size=4096
// CHECK: Removing map entry with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=4096
  // clang-format on
  if (device_alloc != host_alloc)
    printf("a used from device memory with use_device_ptr.\n");

  //
  // map enter/exit + close
  //
  // clang-format off
// CHECK: Entering OpenMP data region with being_mapper at {{.*}} with 1 arguments:
// CHECK: Creating new map entry with HstPtrBase=[[A_HST_PTR]], HstPtrBegin=[[A_HST_PTR]], TgtAllocBegin=[[A_DEV_PTR:0x.*]], TgtPtrBegin=[[A_DEV_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
// CHECK: Copying data from host to device, HstPtr=[[A_HST_PTR]], TgtPtr=[[A_DEV_PTR]], Size=4096
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[A_HST_PTR]]
  // clang-format on
  device_alloc = 0;
#pragma omp target enter data map(close, to : a[ : N])
// clang-format off
// CHECK: Entering OpenMP kernel at {{.*}} with 2 arguments:
// CHECK: Creating new map entry ONLY with HstPtrBase=[[DEVICE_ALLOC_HST_PTR]], HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtAllocBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1, HoldRefCount=0
// CHECK: Mapping exists (implicit) with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=0, DynRefCount=2 (incremented), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=0, DynRefCount=2 (update suppressed), HoldRefCount=0
// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode
// clang-format on
#pragma omp target map(from : device_alloc)
  {
    device_alloc = &a[0];
    a[0] = 99;
  }
  // clang-format off
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=0, DynRefCount=1 (decremented), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Removing map entry with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[A_HST_PTR]]
// CHECK: Exiting OpenMP data region with end_mapper at {{.*}} with 1 arguments:
// CHECK: Mapping exists with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Copying data from device to host, TgtPtr=[[A_DEV_PTR]], HstPtr=[[A_HST_PTR]], Size=4096
// CHECK: Removing map entry with HstPtrBegin=[[A_HST_PTR]], TgtPtrBegin=[[A_DEV_PTR]], Size=4096
  // clang-format on

  // 'close' is missing, so the runtime must check whether s is actually in
  // shared memory in order to determine whether to transfer data and delete the
  // allocation.
#pragma omp target exit data map(from : a[ : N])

  if (device_alloc != host_alloc)
    printf("a has been mapped to the device.\n");

  printf("a[0]=%d\n", a[0]);
  printf("a is present: %d\n", omp_target_is_present(a, dev));

  free(a);

  // CHECK: a used from unified memory.
  // CHECK: a copied to device.
  // CHECK: a used from unified memory with use_device_ptr.

  // CHECK: a used from device memory with use_device_ptr.
  // CHECK: a has been mapped to the device.
  // CHECK: a[0]=99
  // CHECK: a is present: 0

  // CHECK: Done!
  printf("Done!\n");

  return 0;
}
