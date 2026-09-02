// RUN: %libacctarget-compile-run-and-check-generic
// REQUIRES: libacctarget
// REQUIRES: libomptarget-debug

#include <stdio.h>

extern void __tgt_acc_rtl_init();
extern void __tgt_acc_rtl_deinit();

int main(void) {
  // CHECK: Init offload library!
  // CHECK-NOT: Init offload library!
  // CHECK: Deinit offload library!
  // CHECK-NOT: Deinit offload library!
  __tgt_acc_rtl_init();
  __tgt_acc_rtl_init();
  __tgt_acc_rtl_deinit();
  __tgt_acc_rtl_deinit();

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
