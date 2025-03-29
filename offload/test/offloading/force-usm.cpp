// clang-format off
// RUN: %libomptarget-compilexx-generic
// RUN: env LIBOMPTARGET_INFO=32 %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=NO-USM
//
// RUN: %libomptarget-compilexxx-generic-force-usm
// RUN: env HSA_XNACK=1 LIBOMPTARGET_INFO=32 \
// RUN:       %libomptarget-run-generic 2>&1 | %fcheck-generic --check-prefix=FORCE-USM
//
// REQUIRES: unified_shared_memory
//
// UNSUPPORTED: nvptx64-nvidia-cuda
// UNSUPPORTED: nvptx64-nvidia-cuda-LTO
// clang-format on

#include <cassert>
#include <cstdio>
#include <cstdlib>

int GI;
#pragma omp declare target
int *pGI;
#pragma omp end declare target

int main(void) {

  GI = 0;
  // Implicit mappings
  int alpha = 1;
  int beta[3] = {2, 5, 8};

  // Require map clauses for non-USM execution
  pGI = (int *)malloc(sizeof(int));
  *pGI = 42;

#pragma omp target map(pGI[ : 1], GI)
  {
    GI = 1 * alpha;
    *pGI = 2 * beta[1];
  }

  assert(GI == 1);
  assert(*pGI == 10);

  printf("SUCCESS\n");

  return 0;
}

// clang-format off
// NO-USM: omptarget device 0 info: Copying data from host to device, HstPtr={{.*}}, TgtPtr={{.*}}, Size=4
// NO-USM-NEXT: omptarget device 0 info: Copying data from host to device, HstPtr={{.*}}, TgtPtr={{.*}}, Size=12
// NO-USM-NEXT: omptarget device 0 info: Copying data from host to device, HstPtr={{.*}}, TgtPtr={{.*}}, Size=4
// NO-USM-NEXT: omptarget device 0 info: Copying data from host to device, HstPtr={{.*}}, TgtPtr={{.*}}, Size=8, Name=pGI
// NO-USM-NEXT: omptarget device 0 info: Copying data from device to host, TgtPtr={{.*}}, HstPtr={{.*}}, Size=4
// NO-USM-NEXT: omptarget device 0 info: Copying data from device to host, TgtPtr={{.*}}, HstPtr={{.*}}, Size=12
// NO-USM-NEXT: omptarget device 0 info: Copying data from device to host, TgtPtr={{.*}}, HstPtr={{.*}}, Size=4
// NO-USM-NEXT: SUCCESS

// FORCE-USM: SUCCESS
//
// clang-format on
