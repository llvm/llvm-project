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
  int *v = (int *)malloc(sizeof(int) * 100);

// clang-format off
// CHECK: Entering OpenMP kernel at {{.*}} with 1 arguments:
// CHECK: Creating new map entry ONLY with HstPtrBase=[[V_HST_PTR_ADDR:0x.*]], HstPtrBegin=[[V_HST_PTR_ADDR]], TgtAllocBegin=[[V_HST_PTR_ADDR]], TgtPtrBegin=[[V_HST_PTR_ADDR]], Size=200, DynRefCount=1, HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[V_HST_PTR_ADDR]], TgtPtrBegin=[[V_HST_PTR_ADDR]], Size=200, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode

// CHECK: Mapping exists with HstPtrBegin=[[V_HST_PTR_ADDR]], TgtPtrBegin=[[V_HST_PTR_ADDR]], Size=200, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Removing map entry with HstPtrBegin=[[V_HST_PTR_ADDR]], TgtPtrBegin=[[V_HST_PTR_ADDR]], Size=200

// CHECK: Entering OpenMP kernel at {{.*}} with 1 arguments:
// CHECK: Creating new map entry ONLY with HstPtrBase=[[V_HST_PTR_ADDR]], HstPtrBegin=[[V_HST_PTR_ADDR]], TgtAllocBegin=[[V_HST_PTR_ADDR]], TgtPtrBegin=[[V_HST_PTR_ADDR]], Size=280, DynRefCount=1, HoldRefCount=0
// CHECK: Mapping exists with HstPtrBegin=[[V_HST_PTR_ADDR]], TgtPtrBegin=[[V_HST_PTR_ADDR]], Size=280, DynRefCount=1 (update suppressed), HoldRefCount=0
// CHECK: Launching kernel __omp_offloading_{{.*}}_main_l{{.*}} with 1 blocks and 256 threads in Generic mode
// CHECK: Mapping exists with HstPtrBegin=[[V_HST_PTR_ADDR]], TgtPtrBegin=[[V_HST_PTR_ADDR]], Size=280, DynRefCount=0 (decremented, delayed deletion), HoldRefCount=0
// CHECK: Removing map entry with HstPtrBegin=[[V_HST_PTR_ADDR]], TgtPtrBegin=[[V_HST_PTR_ADDR]], Size=280
// clang-format on
#pragma omp target map(tofrom : v[ : 50])
  { v[32] = 32; }

#pragma omp target map(tofrom : v[ : 70])
  { v[64] = 64; }

  printf("v[32] = %d, v[64] = %d\n", v[32], v[64]);

  free(v);

  std::cout << "PASS\n";
  return 0;
}
// CHECK: v[32] = 32, v[64] = 64
// CHECK: PASS
