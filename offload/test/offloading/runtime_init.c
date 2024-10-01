// RUN: %libomptarget-compile-generic
// RUN:   env LIBOMPTARGET_DEBUG=1 %libomptarget-run-generic 2>&1 \
// RUN: %fcheck-generic

// REQUIRES: libomptarget-debug

#include <omp.h>
#include <stdio.h>

extern void __tgt_rtl_init(void);
extern void __tgt_rtl_deinit(void);

// Sanity checks to make sure that this works and is thread safe.
int main() {
  // CHECK: Init offload library!
  // CHECK: Deinit offload library!
  __tgt_rtl_init();
#pragma omp parallel num_threads(8)
  {
    __tgt_rtl_init();
    __tgt_rtl_deinit();
  }
  __tgt_rtl_deinit();

  __tgt_rtl_init();
  __tgt_rtl_deinit();

  // CHECK: PASS
  printf("PASS\n");
}
