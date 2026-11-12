; Test the three-operand forms of XOR.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z196 | FileCheck %s

; Check XRK.
define i32 @f1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f1:
; CHECK: xrk %r2, %r3, %r4
; CHECK: br %r14
  %xor = xor i32 %b, %c
  ret i32 %xor
}

; Check that we can still use XR in obvious cases.
define i32 @f2(i32 %a, i32 %b) {
; CHECK-LABEL: f2:
; CHECK: xr %r2, %r3
; CHECK: br %r14
  %xor = xor i32 %a, %b
  ret i32 %xor
}

; Check XGRK.
define i64 @f3(i64 %a, i64 %b, i64 %c) {
; CHECK-LABEL: f3:
; CHECK: xgrk %r2, %r3, %r4
; CHECK: br %r14
  %xor = xor i64 %b, %c
  ret i64 %xor
}

; Check that we can still use XGR in obvious cases.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: xgr %r2, %r3
; CHECK: br %r14
  %xor = xor i64 %a, %b
  ret i64 %xor
}
