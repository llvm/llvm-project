// RUN: llvm-mc -triple=aarch64-none-linux-gnu -filetype=obj < %s -o - | \
// RUN:   llvm-objdump -r -d - | FileCheck %s

  // TLS add TPREL
  add x2, x1, #:tprel_hi12:var
// CHECK: add x2, x1, #0, lsl #12
// CHECK-NEXT: R_AARCH64_TLSLE_ADD_TPREL_HI12 var

  // TLS add DTPREL
  add x4, x3, #:dtprel_hi12:var
// CHECK: add x4, x3, #0, lsl #12
// CHECK-NEXT: R_AARCH64_TLSLD_ADD_DTPREL_HI12 var
