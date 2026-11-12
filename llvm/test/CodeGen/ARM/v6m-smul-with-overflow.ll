; RUN: llc < %s -mtriple=thumbv6m-none-eabi | FileCheck %s

define i1 @signed_multiplication_did_overflow(i32, i32) {
; CHECK-LABEL: signed_multiplication_did_overflow:
entry-block:
  %2 = tail call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %0, i32 %1)
  %3 = extractvalue { i32, i1 } %2, 1
  ret i1 %3

; CHECK: mov    r2, r1
; CHECK: asrs   r1, r0, #31
; CHECK: asrs   r3, r2, #31
; CHECK: bl     __aeabi_lmul
}

declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32)
