; RUN: llc < %s -mtriple=thumbv6m-none-eabi | FileCheck %s

define i1 @unsigned_multiplication_did_overflow(i32, i32) {
; CHECK-LABEL: unsigned_multiplication_did_overflow:
entry-block:
  %2 = tail call { i32, i1 } @llvm.umul.with.overflow.i32(i32 %0, i32 %1)
  %3 = extractvalue { i32, i1 } %2, 1
  ret i1 %3

; CHECK: mov{{s?}}    r2, r1
; CHECK: mov{{s?}}    r1, #0
; CHECK: mov{{s?}}    r3, {{#0|r1}}
; CHECK: bl     __aeabi_lmul
}

declare { i32, i1 } @llvm.umul.with.overflow.i32(i32, i32)
