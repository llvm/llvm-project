// clang-format off
// RUN: %libomptarget-compilexx-generic && env HSA_XNACK=1 LIBOMPTARGET_INFO=-1 %libomptarget-run-generic 2>&1 | %fcheck-generic
// clang-format on

// REQUIRES: amdgcn-amd-amdhsa
// UNSUPPORTED: clang-6, clang-7, clang-8, clang-9

#include <omp.h>

#include <cassert>
#include <iostream>

#pragma omp requires unified_shared_memory

/// In the current implementation the lack of a target region in the code
/// means that unified shared memory is not being enabled even if the pragma
/// is used explicitly. The code below showcases the copying of data to the
/// GPU.

int main(int argc, char *argv[]) {
  int *v = (int *)malloc(sizeof(int) * 100);

  // clang-format off
// CHECK: Entering OpenMP data region with being_mapper at {{.*}} with 1 arguments:
// CHECK: Creating new map entry with HstPtrBase=[[V_HST_PTR_ADDR:0x.*]], HstPtrBegin=[[V_HST_PTR_ADDR]], TgtAllocBegin=[[V_DEV_PTR_ADDR:0x.*]], TgtPtrBegin=[[V_DEV_PTR_ADDR]], Size=200, DynRefCount=1, HoldRefCount=0
// CHECK: Copying data from host to device, HstPtr=[[V_HST_PTR_ADDR]], TgtPtr=[[V_DEV_PTR_ADDR]], Size=200
  // clang-format on

#pragma omp target enter data map(to : v[ : 50])

  free(v);

  std::cout << "PASS\n";
  return 0;
}
// CHECK: PASS
