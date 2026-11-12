; Test sequences that can use ROSBG.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test the simple case.
define i32 @f1(i32 %a, i32 %b) {
; CHECK-LABEL: f1:
; CHECK: rosbg %r2, %r3, 59, 59, 0
; CHECK: br %r14
  %andb = and i32 %b, 16
  %or = or i32 %a, %andb
  ret i32 %or
}

; ...and again with i64.
define i64 @f2(i64 %a, i64 %b) {
; CHECK-LABEL: f2:
; CHECK: rosbg %r2, %r3, 59, 59, 0
; CHECK: br %r14
  %andb = and i64 %b, 16
  %or = or i64 %a, %andb
  ret i64 %or
}

; Test a case where wraparound is needed.
define i32 @f3(i32 %a, i32 %b) {
; CHECK-LABEL: f3:
; CHECK: rosbg %r2, %r3, 63, 60, 0
; CHECK: br %r14
  %andb = and i32 %b, -7
  %or = or i32 %a, %andb
  ret i32 %or
}

; ...and again with i64.
define i64 @f4(i64 %a, i64 %b) {
; CHECK-LABEL: f4:
; CHECK: rosbg %r2, %r3, 63, 60, 0
; CHECK: br %r14
  %andb = and i64 %b, -7
  %or = or i64 %a, %andb
  ret i64 %or
}

; Test a case with just a shift.
define i32 @f6(i32 %a, i32 %b) {
; CHECK-LABEL: f6:
; CHECK: rosbg %r2, %r3, 32, 51, 12
; CHECK: br %r14
  %shrb = shl i32 %b, 12
  %or = or i32 %a, %shrb
  ret i32 %or
}

; ...and again with i64.
define i64 @f7(i64 %a, i64 %b) {
; CHECK-LABEL: f7:
; CHECK: rosbg %r2, %r3, 0, 51, 12
; CHECK: br %r14
  %shrb = shl i64 %b, 12
  %or = or i64 %a, %shrb
  ret i64 %or
}

; Test a case with just a rotate.  This can't use ROSBG.
define i32 @f8(i32 %a, i32 %b) {
; CHECK-LABEL: f8:
; CHECK: rll {{%r[0-5]}}
; CHECK: or {{%r[0-5]}}
; CHECK: br %r14
  %shlb = shl i32 %b, 30
  %shrb = lshr i32 %b, 2
  %rotlb = or i32 %shlb, %shrb
  %or = or i32 %a, %rotlb
  ret i32 %or
}

; ...and again with i64, which can.
define i64 @f9(i64 %a, i64 %b) {
; CHECK-LABEL: f9:
; CHECK: rosbg %r2, %r3, 0, 63, 47
; CHECK: br %r14
  %shlb = shl i64 %b, 47
  %shrb = lshr i64 %b, 17
  %rotlb = or i64 %shlb, %shrb
  %or = or i64 %a, %rotlb
  ret i64 %or
}

; Test a case with a shift and AND.
define i32 @f10(i32 %a, i32 %b) {
; CHECK-LABEL: f10:
; CHECK: rosbg %r2, %r3, 56, 59, 4
; CHECK: br %r14
  %shrb = shl i32 %b, 4
  %andb = and i32 %shrb, 240
  %or = or i32 %a, %andb
  ret i32 %or
}

; ...and again with i64.
define i64 @f11(i64 %a, i64 %b) {
; CHECK-LABEL: f11:
; CHECK: rosbg %r2, %r3, 56, 59, 4
; CHECK: br %r14
  %shrb = shl i64 %b, 4
  %andb = and i64 %shrb, 240
  %or = or i64 %a, %andb
  ret i64 %or
}

; Check the handling of zext and OR, which can use ROSBG.
define i64 @f12(i64 %a, i32 %b) {
; CHECK-LABEL: f12:
; CHECK: rosbg %r2, %r3, 32, 63, 0
; CHECK: br %r14
  %add = add i32 %b, 1
  %ext = zext i32 %add to i64
  %or = or i64 %a, %ext
  ret i64 %or
}
