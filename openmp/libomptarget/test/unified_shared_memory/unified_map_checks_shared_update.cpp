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

// clang-format off
// CHECK: Creating new map entry ONLY with HstPtrBase=[[DEVICE_DATA_HST_PTR:0x.*]], HstPtrBegin=[[DEVICE_DATA_HST_PTR]], TgtAllocBegin=[[DEVICE_DATA_HST_PTR]], TgtPtrBegin=[[DEVICE_DATA_HST_PTR]], Size=8, DynRefCount=1, HoldRefCount=0
// CHECK: Creating new map entry ONLY with HstPtrBase=[[DATA_HST_PTR:0x.*]], HstPtrBegin=[[DATA_HST_PTR]], TgtAllocBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_HST_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0
// CHECK: Creating new map entry ONLY with HstPtrBase=[[DEVICE_ALLOC_HST_PTR:0x.*]], HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtAllocBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1, HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_DATA_HST_PTR]], TgtPtrBegin=[[DEVICE_DATA_HST_PTR]], Size=8, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_HST_PTR]], Size=4096, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=1 (update suppressed), HoldRefCount=0

// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode

// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]], TgtPtrBegin=[[DEVICE_ALLOC_HST_PTR]], Size=8, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_HST_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[DEVICE_DATA_HST_PTR]], TgtPtrBegin=[[DEVICE_DATA_HST_PTR]], Size=8, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0

// CHECK: Removing map entry with HstPtrBegin=[[DEVICE_ALLOC_HST_PTR]]{{.*}} Size=8
// CHECK: Removing map entry with HstPtrBegin=[[DATA_HST_PTR]]{{.*}} Size=4096
// CHECK: Removing map entry with HstPtrBegin=[[DEVICE_DATA_HST_PTR]]{{.*}} Size=8
// clang-format on

// implicit mapping of data
#pragma omp target map(tofrom : device_data, device_alloc)
  {
    device_data = &data[0];
    device_alloc = &alloc[0];

    for (int i = 0; i < N; i++) {
      alloc[i] += 1;
      data[i] += 1;
    }
  }

  if (device_alloc == host_alloc)
    printf("Address of alloc on device matches host address.\n");

  if (device_data == host_data)
    printf("Address of data on device matches host address.\n");

  // On the host, check that the arrays have been updated.
  fails = 0;
  for (int i = 0; i < N; i++) {
    if (alloc[i] != 11)
      fails++;
  }
  printf("Alloc device values updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  fails = 0;
  for (int i = 0; i < N; i++) {
    if (data[i] != 2)
      fails++;
  }
  printf("Data device values updated: %s\n",
         (fails == 0) ? "Succeeded" : "Failed");

  //
  // Test that updates on the host and on the device are both visible.
  //

  // Update on the host.
  for (int i = 0; i < N; ++i) {
    alloc[i] += 1;
    data[i] += 1;
  }

  // clang-format off
// CHECK: Creating new map entry ONLY with HstPtrBase=[[DATA_HST_PTR]], HstPtrBegin=[[DATA_HST_PTR]], TgtAllocBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_HST_PTR]], Size=4096, DynRefCount=1, HoldRefCount=0

// CHECK: Mapping exists with HstPtrBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_HST_PTR]], Size=4096, DynRefCount=1 (update suppressed), HoldRefCount=0

// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode

// CHECK: Mapping exists with HstPtrBegin=[[DATA_HST_PTR]], TgtPtrBegin=[[DATA_HST_PTR]], Size=4096, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0

// CHECK: Removing map entry with HstPtrBegin=[[DATA_HST_PTR]]{{.*}} Size=4096
  // clang-format on

  int alloc_fails = 0;
  int data_fails = 0;
#pragma omp target
  {
    for (int i = 0; i < N; i++) {
      if (alloc[i] != 12)
        alloc_fails++;
    }
    for (int i = 0; i < N; i++) {
      if (data[i] != 3)
        data_fails++;
    }
  }
  printf("Alloc host values updated: %s\n",
         (alloc_fails == 0) ? "Succeeded" : "Failed");
  printf("Data host values updated: %s\n",
         (data_fails == 0) ? "Succeeded" : "Failed");
  free(alloc);

  // CHECK: Address of alloc on device matches host address.
  // CHECK: Address of data on device matches host address.
  // CHECK: Alloc device values updated: Succeeded
  // CHECK: Data device values updated: Succeeded

  // CHECK: Alloc host values updated: Succeeded
  // CHECK: Data host values updated: Succeeded

  printf("Done!\n");

  return 0;
}
