// clang-format off
// RUN: %libomptarget-compilexx-generic -lhsa-runtime64
// RUN: env HSA_XNACK=1 OMPX_APU_MAPS=1 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=ENABLED -check-prefix=CHECK
// RUN: env HSA_XNACK=0 OMPX_APU_MAPS=0 \
// RUN:   %libomptarget-run-generic 2>&1 | %fcheck-generic -check-prefix=DISABLED -check-prefix=CHECK
// clang-format on

// REQUIRES: amdgcn-amd-amdhsa
// REQUIRES: hsa

#include "hsa.h"
#include "hsa_ext_amd.h"
#include <iostream>

void f() {
  bool hasSystemXnackEnabled = false;
  hsa_status_t HsaStatus = hsa_system_get_info(
      HSA_AMD_SYSTEM_INFO_XNACK_ENABLED, &hasSystemXnackEnabled);
  if (HsaStatus != HSA_STATUS_SUCCESS)
    printf("Output status is bad!\n");
  printf("hasSystemXnackEnabled = %d\n", hasSystemXnackEnabled);
}

int main() {
  // CHECK-NOT: Output status is bad!
  // ENABLED: hasSystemXnackEnabled = 1
  // DISABLED: hasSystemXnackEnabled = 0
  f();
  return 0;
}
