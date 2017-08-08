; Test shortening of NILL to NILF when the result is used as a rotate amount.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test 32-bit rotate.
define i32 @f1(i32 %val, i32 %amt) {
; CHECK-LABEL: f1:
; CHECK: nill %r3, 31
; CHECK: rll %r2, %r2, 0(%r3)
  %mod = urem i32 %amt, 32

  %inv = sub i32 32, %mod
  %parta = shl i32 %val, %mod
  %partb = lshr i32 %val, %inv

  %rotl = or i32 %parta, %partb

  ret i32 %rotl
}

; Test 64-bit rotate.
define i64 @f2(i64 %val, i64 %amt) {
; CHECK-LABEL: f2:
; CHECK: nill %r3, 31
; CHECK: rllg %r2, %r2, 0(%r3)
  %mod = urem i64 %amt, 32

  %inv = sub i64 64, %mod
  %parta = shl i64 %val, %mod
  %partb = lshr i64 %val, %inv

  %rotl = or i64 %parta, %partb

  ret i64 %rotl
}
