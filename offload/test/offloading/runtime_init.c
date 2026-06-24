// RUN: %clang %flags %flags_clang -lomptarget %s -o %t.a.out
// RUN:   env LIBOMPTARGET_DEBUG=1 %t.a.out 2>&1 | \
// RUN: %fcheck-generic

// REQUIRES: libomptarget-debug

#include <omp.h>
#include <stdbool.h>
#include <stdio.h>

extern void __tgt_rtl_init(bool OffloadEnabled);
extern void __tgt_rtl_deinit(void);

// Sanity checks to make sure that this works and is thread safe.
int main() {
  // CHECK: Init offload library!
  // CHECK: Offload is disabled. Skipping plugin initialization
  __tgt_rtl_init(false);
#pragma omp parallel num_threads(8)
  {
    __tgt_rtl_init(false);
    __tgt_rtl_deinit();
  }
  // CHECK-NOT: Offload is enabled. Initializating plugins
  // CHECK: Deinit offload library!
  __tgt_rtl_deinit();

  // CHECK: Init offload library!
  // CHECK: Offload is enabled. Initializating plugins
  __tgt_rtl_init(true);
  // CHECK: Deinit offload library!
  __tgt_rtl_deinit();

  // CHECK: Init offload library!
  // CHECK: Offload is disabled. Skipping plugin initialization
  __tgt_rtl_init(false);
  // CHECK: Offload is enabled. Initializating plugins
  __tgt_rtl_init(true);
  __tgt_rtl_deinit();
  // CHECK: Deinit offload library!
  __tgt_rtl_deinit();

  // CHECK: Init offload library!
  // CHECK: Offload is enabled. Initializating plugins
  __tgt_rtl_init(true);
  // CHECK-NOT: Offload is disabled. Skipping plugin initialization
  __tgt_rtl_init(false);
  __tgt_rtl_deinit();
  // CHECK: Deinit offload library!
  __tgt_rtl_deinit();

  // CHECK-NOT: Init offload library!
  // CHECK-NOT: Deinit offload library!

  // CHECK: PASS
  printf("PASS\n");
}
