; Test compound shifts.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test a shift right followed by a sign extension.  This can use two shifts.
define i64 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: risbg %r0, %r2, 63, 191, 63
; CHECK: lcgr  %r2, %r0
; CHECK: br %r14
  %shr = lshr i32 %a, 1
  %trunc = trunc i32 %shr to i1
  %ext = sext i1 %trunc to i64
  ret i64 %ext
}

; ...and again with the highest shift count that doesn't reduce to an
; ashr/sext pair.
define i64 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: risbg %r0, %r2, 63, 191, 34
; CHECK: lcgr  %r2, %r0
; CHECK: br %r14
  %shr = lshr i32 %a, 30
  %trunc = trunc i32 %shr to i1
  %ext = sext i1 %trunc to i64
  ret i64 %ext
}

; Test a left shift that of an extended right shift in a case where folding
; is possible.
define i64 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: risbg %r2, %r2, 27, 181, 9
; CHECK: br %r14
  %shr = lshr i32 %a, 1
  %ext = zext i32 %shr to i64
  %shl = shl i64 %ext, 10
  %and = and i64 %shl, 137438952960
  ret i64 %and
}

; ...and again with a larger right shift.
define i64 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: risbg %r2, %r2, 30, 158, 3
; CHECK: br %r14
  %shr = lshr i32 %a, 30
  %ext = sext i32 %shr to i64
  %shl = shl i64 %ext, 33
  %and = and i64 %shl, 8589934592
  ret i64 %and
}

; Repeat the previous test in a case where all bits outside the
; bottom 3 matter.
define i64 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: risbg %r2, %r2, 29, 158, 3
; CHECK: lhi %r2, 7
; CHECK: br %r14
  %shr = lshr i32 %a, 30
  %ext = sext i32 %shr to i64
  %shl = shl i64 %ext, 33
  %or = or i64 %shl, 7
  ret i64 %or
}

; Test that SRA gets replaced with SRL if the sign bit is the only one
; that matters.
define i64 @f6(i64 %a) {
; CHECK-LABEL: f6:
; CHECK: risbg %r2, %r2, 55, 183, 19
; CHECK: br %r14
  %shl = shl i64 %a, 10
  %shr = ashr i64 %shl, 60
  %and = and i64 %shr, 256
  ret i64 %and
}

; Test another form of f1.
define i64 @f7(i32 %a) {
; CHECK-LABEL: f7:
; CHECK: sllg [[REG:%r[0-5]]], %r2, 62
; CHECK: srag %r2, [[REG]], 63
; CHECK: br %r14
  %1 = shl i32 %a, 30
  %sext = ashr i32 %1, 31
  %ext = sext i32 %sext to i64
  ret i64 %ext
}
