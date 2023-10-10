// clang-format off
// RUN: %libomptarget-compilexx-generic && env HSA_XNACK=1 LIBOMPTARGET_INFO=-1 %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// REQUIRES: amdgcn-amd-amdhsa
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

#include <omp.h>

#include <cassert>
#include <iostream>

#pragma omp requires unified_shared_memory

int main(int argc, char *argv[]) {
  int x = 5;
  int y = 7;
  int z = 11;
  int *v = (int *)malloc(sizeof(int) * 10);
// clang-format off
// CHECK: Entering OpenMP data region with being_mapper at {{.*}} with 1 arguments:
// CHECK: Creating new map entry ONLY with HstPtrBase=[[Z_HST_PTR_BEGIN:0x.*]], HstPtrBegin=[[Z_HST_PTR_BEGIN]], TgtAllocBegin=[[Z_HST_PTR_BEGIN]], TgtPtrBegin=[[Z_HST_PTR_BEGIN]], Size=4, DynRefCount=1, HoldRefCount=0
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[Z_HST_PTR_BEGIN]]{{.*}}[[Z_HST_PTR_BEGIN]]
// clang-format on
#pragma omp target enter data map(to : z)
// clang-format off
// CHECK: Entering OpenMP kernel at {{.*}} with 4 arguments:
// CHECK: Creating new map entry ONLY with HstPtrBase=[[X_HST_PTR_BEGIN:0x.*]], HstPtrBegin=[[X_HST_PTR_BEGIN]], TgtAllocBegin=[[X_HST_PTR_BEGIN]], TgtPtrBegin=[[X_HST_PTR_BEGIN]], Size=4, DynRefCount=1, HoldRefCount=0
// CHECK: Creating new map entry ONLY with HstPtrBase=[[Y_HST_PTR_BEGIN:0x.*]], HstPtrBegin=[[Y_HST_PTR_BEGIN]], TgtAllocBegin=[[Y_HST_PTR_BEGIN]], TgtPtrBegin=[[Y_HST_PTR_BEGIN]], Size=4, DynRefCount=1, HoldRefCount=0
// CHECK: variable {{.*}} does not have a valid device counterpart
// CHECK: Mapping exists with HstPtrBegin=[[X_HST_PTR_BEGIN]]{{.*}} Size=4, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[Y_HST_PTR_BEGIN]]{{.*}} Size=4, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode
// clang-format on
#pragma omp target map(tofrom : x) map(always, tofrom : y) map(to : v[ : 0])
  {
    x++;
    y++;
    z++;
  }

  // clang-format off
// CHECK: Mapping exists with HstPtrBegin=[[Y_HST_PTR_BEGIN]]{{.*}} Size=4, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[X_HST_PTR_BEGIN]]{{.*}} Size=4, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Removing map entry with HstPtrBegin=[[Y_HST_PTR_BEGIN]], TgtPtrBegin=[[Y_HST_PTR_BEGIN]], Size=4
// CHECK: Removing map entry with HstPtrBegin=[[X_HST_PTR_BEGIN]], TgtPtrBegin=[[X_HST_PTR_BEGIN]], Size=4
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[Z_HST_PTR_BEGIN]]{{.*}}[[Z_HST_PTR_BEGIN]]
  // clang-format on
  printf("x = %d, y = %d, z = %d\n", x, y, z);

// clang-format off
// CHECK: Entering OpenMP kernel at {{.*}} with 4 arguments:
// CHECK: Creating new map entry ONLY with HstPtrBase=[[X_HST_PTR_BEGIN:0x.*]], HstPtrBegin=[[X_HST_PTR_BEGIN]], TgtAllocBegin=[[X_HST_PTR_BEGIN]], TgtPtrBegin=[[X_HST_PTR_BEGIN]], Size=4, DynRefCount=1, HoldRefCount=0
// CHECK: Creating new map entry ONLY with HstPtrBase=[[Y_HST_PTR_BEGIN:0x.*]], HstPtrBegin=[[Y_HST_PTR_BEGIN]], TgtAllocBegin=[[Y_HST_PTR_BEGIN]], TgtPtrBegin=[[Y_HST_PTR_BEGIN]], Size=4, DynRefCount=1, HoldRefCount=0
// CHECK: variable {{.*}} does not have a valid device counterpart
// CHECK: Mapping exists with HstPtrBegin=[[X_HST_PTR_BEGIN]]{{.*}} Size=4, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[Y_HST_PTR_BEGIN]]{{.*}} Size=4, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode
// clang-format on
#pragma omp target map(tofrom : x) map(always, tofrom : y) map(to : v[ : 0])
  {
    x++;
    y++;
    z++;
  }
// clang-format off
// CHECK: Mapping exists with HstPtrBegin=[[Y_HST_PTR_BEGIN]]{{.*}} Size=4, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[X_HST_PTR_BEGIN]]{{.*}} Size=4, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Removing map entry with HstPtrBegin=[[Y_HST_PTR_BEGIN]], TgtPtrBegin=[[Y_HST_PTR_BEGIN]], Size=4
// CHECK: Removing map entry with HstPtrBegin=[[X_HST_PTR_BEGIN]], TgtPtrBegin=[[X_HST_PTR_BEGIN]], Size=4
// CHECK: OpenMP Host-Device pointer mappings after block
// CHECK: Host Ptr
// CHECK: [[Z_HST_PTR_BEGIN]]{{.*}}[[Z_HST_PTR_BEGIN]]
// clang-format on
#pragma omp target exit data map(from : z)
  // clang-format off
// CHECK: Exiting OpenMP data region with end_mapper at {{.*}} with 1 arguments:
// CHECK: Mapping exists with HstPtrBegin=[[Z_HST_PTR_BEGIN]], TgtPtrBegin=[[Z_HST_PTR_BEGIN]], Size=4, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Removing map entry with HstPtrBegin=[[Z_HST_PTR_BEGIN]], TgtPtrBegin=[[Z_HST_PTR_BEGIN]], Size=4
  // clang-format on
  printf("x = %d, y = %d, z = %d\n", x, y, z);

  free(v);

  std::cout << "PASS\n";
  return 0;
}
// CHECK: x = 6, y = 8, z = 11
// CHECK: x = 7, y = 9, z = 11
// CHECK: PASS
