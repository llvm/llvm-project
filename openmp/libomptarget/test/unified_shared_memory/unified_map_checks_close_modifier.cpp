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
  int fails;
  void *host_alloc = nullptr, *device_alloc = nullptr;
  void *host_data = nullptr, *device_data = nullptr;
  int *alloc = (int *)malloc(N * sizeof(int));
  int data[N];

  for (int i = 0; i < N; ++i) {
    alloc[i] = 10;
    data[i] = 1;
  }

  host_data = &data[0];
  host_alloc = &alloc[0];

  //
  // Test that updates on the device are not visible to host
  // when only a TO mapping is used.
  //

  // clang-format off
// CHECK: Creating new map entry ONLY with HstPtrBase=[[DEVICE_DATA_HST_PTR:0x.*]], HstPtrBegin=[[DEVICE_DATA_HST_PTR]], TgtAllocBegin=[[DEVICE_DATA_HST_PTR]], TgtPtrBegin=[[DEVICE_DATA_HST_PTR]], Size=8, DynRefCount=1, HoldRefCount=0
// CHECK: Creating new map entry with HstPtrBase=[[DATA_HST_PTR:0x.*]], HstPtrBegin=[[DATA_HST_PTR]], TgtAllocBegin=[[DATA_DEV_PTR:0x.*]], TgtPtrBegin=[[DATA_DEV_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
// CHECK: Copying data from host to device, HstPtr=[[DATA_HST_PTR]], TgtPtr=[[DATA_DEV_PTR]], Size=4096
// CHECK: Creating new map entry ONLY with HstPtrBase=[[DEVICE_ALLOC_HST_PTR:0x.*]], HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtAllocBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1, HoldRefCount=0
// CHECK: Creating new map entry with HstPtrBase=[[ALLOC_HST_PTR:0x.*]], HstPtrBegin=[[ALLOC_HST_PTR]], TgtAllocBegin=[[ALLOC_DEV_PTR:0x.*]], TgtPtrBegin=[[ALLOC_DEV_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
// CHECK: Copying data from host to device, HstPtr=[[ALLOC_HST_PTR]], TgtPtr=[[ALLOC_DEV_PTR]], Size=4096

// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_DATA_HST_PTR]], TgtPtrBegin=[[DEVICE_DATA_HST_PTR]], Size=8, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_DEV_PTR]], Size=4096, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[ALLOC_HST_PTR]], TgtPtrBegin=[[ALLOC_DEV_PTR]], Size=4096, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode

// CHECK: Mapping exists with HstPtrBegin=[[ALLOC_HST_PTR]], TgtPtrBegin=[[ALLOC_DEV_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_DEV_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_DATA_HST_PTR]], TgtPtrBegin=[[DEVICE_DATA_HST_PTR]], Size=8, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0

// CHECK: Removing map entry with HstPtrBegin=[[ALLOC_HST_PTR]]{{.*}} Size=4096
// CHECK: Removing map entry with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]]{{.*}} Size=8
// CHECK: Removing map entry with HstPtrBegin=[[DATA_HST_PTR]]{{.*}} Size=4096
// CHECK: Removing map entry with HstPtrBegin=[[DEVICE_DATA_HST_PTR]]{{.*}} Size=8
  // clang-format on

#pragma omp target map(tofrom : device_data, device_alloc)                     \
    map(close, to : alloc[ : N], data[ : N])
  {
    device_data = &data[0];
    device_alloc = &alloc[0];

    for (int i = 0; i < N; i++) {
      alloc[i] += 1;
      data[i] += 1;
    }
  }

  if (device_alloc != host_alloc)
    printf("Address of alloc on device different from host address.\n");

  if (device_data != host_data)
    printf("Address of data on device different from host address.\n");

  // On the host, check that the arrays have been updated.
  fails = 0;
  for (int i = 0; i < N; i++) {
    if (alloc[i] != 10)
      fails++;
  }
  printf("Alloc host values not updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  fails = 0;
  for (int i = 0; i < N; i++) {
    if (data[i] != 1)
      fails++;
  }
  printf("Data host values not updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  //
  // Test that updates on the device are visible on host
  // when a from is used.
  //

  for (int i = 0; i < N; i++) {
    alloc[i] += 1;
    data[i] += 1;
  }

  // clang-format off
  // CHECK: Creating new map entry with HstPtrBase=[[ALLOC_HST_PTR:0x.*]], HstPtrBegin=[[ALLOC_HST_PTR]], TgtAllocBegin=[[ALLOC_DEV_PTR:0x.*]], TgtPtrBegin=[[ALLOC_DEV_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
  // CHECK: Copying data from host to device, HstPtr=[[ALLOC_HST_PTR]], TgtPtr=[[ALLOC_DEV_PTR]], Size=4096

  // CHECK: Creating new map entry with HstPtrBase=[[DATA_HST_PTR:0x.*]], HstPtrBegin=[[DATA_HST_PTR]], TgtAllocBegin=[[DATA_DEV_PTR:0x.*]], TgtPtrBegin=[[DATA_DEV_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
  // CHECK: Copying data from host to device, HstPtr=[[DATA_HST_PTR]], TgtPtr=[[DATA_DEV_PTR]], Size=4096

  // CHECK: Mapping exists with HstPtrBegin=[[ALLOC_HST_PTR]], TgtPtrBegin=[[ALLOC_DEV_PTR]], Size=4096, DynRefCount=1 (update suppressed), HoldRefCount=0
  // CHECK: Mapping exists with HstPtrBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_DEV_PTR]], Size=4096, DynRefCount=1 (update suppressed), HoldRefCount=0

  // CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode

  // CHECK: Mapping exists with HstPtrBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_DEV_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
  // CHECK: Mapping exists with HstPtrBegin=[[ALLOC_HST_PTR]], TgtPtrBegin=[[ALLOC_DEV_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0

  // CHECK: Removing map entry with HstPtrBegin=[[DATA_HST_PTR]]{{.*}} Size=4096
  // CHECK: Removing map entry with HstPtrBegin=[[ALLOC_HST_PTR]]{{.*}} Size=4096
  // clang-format on

  int alloc_fails = 0;
  int data_fails = 0;
#pragma omp target map(close, tofrom : alloc[ : N], data[ : N])                \
    map(tofrom : alloc_fails, data_fails)
  {
    for (int i = 0; i < N; i++) {
      if (alloc[i] != 11)
        alloc_fails++;
    }
    for (int i = 0; i < N; i++) {
      if (data[i] != 2)
        data_fails++;
    }

    // Update values on the device
    for (int i = 0; i < N; i++) {
      alloc[i] += 1;
      data[i] += 1;
    }
  }

  printf("Alloc device values are correct: %s\n",
         (alloc_fails == 0) ? "Succeeded" : "Failed");
  printf("Data device values are correct: %s\n",
         (data_fails == 0) ? "Succeeded" : "Failed");

  fails = 0;
  for (int i = 0; i < N; i++) {
    if (alloc[i] != 12)
      fails++;
  }
  printf("Alloc host values updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  fails = 0;
  for (int i = 0; i < N; i++) {
    if (data[i] != 3)
      fails++;
  }
  printf("Data host values updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  free(alloc);

  // CHECK: Address of alloc on device different from host address.
  // CHECK: Address of data on device different from host address.
  // On the host, check that the arrays have been updated.
  // CHECK: Alloc host values not updated: Succeeded
  // CHECK: Data host values not updated: Succeeded

  // CHECK: Alloc device values are correct: Succeeded
  // CHECK: Data device values are correct: Succeeded
  // CHECK: Alloc host values updated: Succeeded
  // CHECK: Data host values updated: Succeeded

  // CHECK: Done!
  printf("Done!\n");

  return 0;
}
