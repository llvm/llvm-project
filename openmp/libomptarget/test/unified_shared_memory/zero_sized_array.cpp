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
  int *v = (int *)malloc(sizeof(int) * 10);

  printf("host address of v = %p\n", v);

// CHECK: variable {{.*}} does not have a valid device counterpart
#pragma omp target map(to : v[ : 0])
  { printf("device address of v = %p\n", v); }

  free(v);

  std::cout << "PASS\n";
  return 0;
}
// CHECK: host address of v = [[ADDR_OF_V:0x.*]]
// TODO: once printf is supported add check for ADDR_OF_V on device
// CHECK: PASS
