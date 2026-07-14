; RUN: opt < %s -passes=constraint-elimination -S | FileCheck %s

; ConstraintElimination models `lshr x, n <= x` (a logical right shift never
; increases an unsigned value), like it already does for udiv/urem. The fact is
; only added when the shift is guaranteed not to be poison (shift < bitwidth).

define i1 @lshr_in_bounds(i64 %x, i64 %n) {
; CHECK-LABEL: @lshr_in_bounds(
; CHECK:       guarded:
; CHECK-NEXT:    [[H:%.*]] = lshr i64 [[X:%.*]], 1
; CHECK-NEXT:    ret i1 true
entry:
  %inrange = icmp ult i64 %x, %n
  br i1 %inrange, label %guarded, label %out
guarded:
  %h = lshr i64 %x, 1
  %chk = icmp ult i64 %h, %n
  ret i1 %chk
out:
  ret i1 false
}

; Negative: shift amount == bitwidth makes `lshr` poison, so no fact is added
; and the check must be preserved.
define i1 @lshr_oob_const(i64 %x, i64 %n) {
; CHECK-LABEL: @lshr_oob_const(
; CHECK:       guarded:
; CHECK-NEXT:    [[H:%.*]] = lshr i64 [[X:%.*]], 64
; CHECK-NEXT:    [[CHK:%.*]] = icmp ult i64 [[H]], [[N:%.*]]
; CHECK-NEXT:    ret i1 [[CHK]]
entry:
  %inrange = icmp ult i64 %x, %n
  br i1 %inrange, label %guarded, label %out
guarded:
  %h = lshr i64 %x, 64
  %chk = icmp ult i64 %h, %n
  ret i1 %chk
out:
  ret i1 false
}

; Negative: a variable shift may be >= bitwidth (poison), so no fact is added
; and the check must be preserved.
define i1 @lshr_var_shift(i64 %x, i64 %n, i64 %k) {
; CHECK-LABEL: @lshr_var_shift(
; CHECK:       guarded:
; CHECK-NEXT:    [[H:%.*]] = lshr i64 [[X:%.*]], [[K:%.*]]
; CHECK-NEXT:    [[CHK:%.*]] = icmp ult i64 [[H]], [[N:%.*]]
; CHECK-NEXT:    ret i1 [[CHK]]
entry:
  %inrange = icmp ult i64 %x, %n
  br i1 %inrange, label %guarded, label %out
guarded:
  %h = lshr i64 %x, %k
  %chk = icmp ult i64 %h, %n
  ret i1 %chk
out:
  ret i1 false
}
