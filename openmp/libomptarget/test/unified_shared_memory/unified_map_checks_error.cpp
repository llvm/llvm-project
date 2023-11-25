// clang-format off
// RUN: %libomptarget-compilexx-generic && env HSA_XNACK=1 LIBOMPTARGET_INFO=-1 %libomptarget-run-fail-generic 2>&1 | %fcheck-generic
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
// CHECK: Entering OpenMP data region with being_mapper at {{.*}} with 1 arguments:
// CHECK: Creating new map entry ONLY with HstPtrBase=[[V_HST_PTR_ADDR:0x.*]], HstPtrBegin=[[V_HST_PTR_ADDR]], TgtAllocBegin=[[V_HST_PTR_ADDR]], TgtPtrBegin=[[V_HST_PTR_ADDR]], Size=200, DynRefCount=1, HoldRefCount=0
// CHECK: explicit extension not allowed: host address specified is [[V_HST_PTR_ADDR]] (280 bytes), but device allocation maps to host at [[V_HST_PTR_ADDR]] (200 bytes)
// CHECK: Call to getTargetPointer returned null pointer (device failure or illegal mapping).
// clang-format on
#pragma omp target enter data map(to : v[ : 50])

#pragma omp target enter data map(to : v[ : 70])

#pragma omp target
  {}

  free(v);

  std::cout << "PASS\n";
  return 0;
}
// CHECK-NOT: PASS
