; Test sequences that can use RXSBG.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test the simple case.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: rxsbg %r2, %r3, 59, 59, 0
; CHECK: br %r14
  %andb = and i32 %b, 16
  %xor = xor i32 %a, %andb
  ret i32 %xor
}

; ...and again with i64.
define i64 @f2(i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: rxsbg %r2, %r3, 59, 59, 0
; CHECK: br %r14
  %andb = and i64 %b, 16
  %xor = xor i64 %a, %andb
  ret i64 %xor
}

; Test a case where wraparound is needed.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: rxsbg %r2, %r3, 63, 60, 0
; CHECK: br %r14
  %andb = and i32 %b, -7
  %xor = xor i32 %a, %andb
  ret i32 %xor
}

; ...and again with i64.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: rxsbg %r2, %r3, 63, 60, 0
; CHECK: br %r14
  %andb = and i64 %b, -7
  %xor = xor i64 %a, %andb
  ret i64 %xor
}

; Test a case with just a shift.
define i32 @f6(i32 %a, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: rxsbg %r2, %r3, 32, 51, 12
; CHECK: br %r14
  %shlb = shl i32 %b, 12
  %xor = xor i32 %a, %shlb
  ret i32 %xor
}

; ...and again with i64.
define i64 @f7(i64 %a, i64 %b) {
; CHECK-LABEL: f7:
; CHECK: rxsbg %r2, %r3, 0, 51, 12
; CHECK: br %r14
  %shlb = shl i64 %b, 12
  %xor = xor i64 %a, %shlb
  ret i64 %xor
}

; Test a case with just a rotate (using XOR for the rotate combination too,
; to test that this kind of rotate does get recognised by the target-
; independent code).  This can't use RXSBG.
define i32 @f8(i32 %a, i32 %b) {
; CHECK-LABEL: f8:
; CHECK: rll {{%r[0-5]}}
; CHECK: xr {{%r[0-5]}}
; CHECK: br %r14
  %shlb = shl i32 %b, 30
  %shrb = lshr i32 %b, 2
  %rotlb = xor i32 %shlb, %shrb
  %xor = xor i32 %a, %rotlb
  ret i32 %xor
}

; ...and again with i64, which can use RXSBG for the rotate.
define i64 @f9(i64 %a, i64 %b) {
; CHECK-LABEL: f9:
; CHECK: rxsbg %r2, %r3, 0, 63, 47
; CHECK: br %r14
  %shlb = shl i64 %b, 47
  %shrb = lshr i64 %b, 17
  %rotlb = xor i64 %shlb, %shrb
  %xor = xor i64 %a, %rotlb
  ret i64 %xor
}

; Test a case with a shift and AND.
define i32 @f10(i32 %a, i32 %b) {
; CHECK-LABEL: f10:
; CHECK: rxsbg %r2, %r3, 56, 59, 4
; CHECK: br %r14
  %shlb = shl i32 %b, 4
  %andb = and i32 %shlb, 240
  %xor = xor i32 %a, %andb
  ret i32 %xor
}

; ...and again with i64.
define i64 @f11(i64 %a, i64 %b) {
; CHECK-LABEL: f11:
; CHECK: rxsbg %r2, %r3, 56, 59, 4
; CHECK: br %r14
  %shlb = shl i64 %b, 4
  %andb = and i64 %shlb, 240
  %xor = xor i64 %a, %andb
  ret i64 %xor
}

; Check the handling of zext and XOR, which can use ROSBG.
define i64 @f12(i64 %a, i32 %b) {
; CHECK-LABEL: f12:
; CHECK: rxsbg %r2, %r3, 32, 63, 0
; CHECK: br %r14
  %add = add i32 %b, 1
  %ext = zext i32 %add to i64
  %xor = xor i64 %a, %ext
  ret i64 %xor
}
