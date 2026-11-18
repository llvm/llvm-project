// RUN: %libacctarget-compile-generic && env LIBOMPTARGET_DEBUG=1 %libacctarget-run-generic 2>&1 | %fcheck-generic
// REQUIRES: libacctarget, libomptarget-debug

#include <stdio.h>

extern void __tgt_acc_rtl_init();
extern void __tgt_acc_rtl_deinit();

int main(void) {
  // CHECK: Init OpenACC runtime
  // CHECK-NOT: Init OpenACC runtime
  // CHECK: Deinit OpenACC runtime
  // CHECK-NOT: Deinit OpenACC runtime
  __tgt_acc_rtl_init();
  __tgt_acc_rtl_init();
  __tgt_acc_rtl_deinit();
  __tgt_acc_rtl_deinit();

  // CHECK: PASS
  printf("PASS\n");
  return 0;
}
