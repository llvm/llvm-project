; Test shortening of NILL to NILF when the result is used as a shift amount.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test logical shift right.
define i32 @f1(i32 %a, i32 %sh) {
; CHECK-LABEL: f1:
; CHECK: nill %r3, 31
; CHECK: srl %r2, 0(%r3)
  %and = and i32 %sh, 31
  %shift = lshr i32 %a, %and
  ret i32 %shift
}

; Test arithmetic shift right.
define i32 @f2(i32 %a, i32 %sh) {
; CHECK-LABEL: f2:
; CHECK: nill %r3, 31
; CHECK: sra %r2, 0(%r3)
  %and = and i32 %sh, 31
  %shift = ashr i32 %a, %and
  ret i32 %shift
}

; Test shift left.
define i32 @f3(i32 %a, i32 %sh) {
; CHECK-LABEL: f3:
; CHECK: nill %r3, 31
; CHECK: sll %r2, 0(%r3)
  %and = and i32 %sh, 31
  %shift = shl i32 %a, %and
  ret i32 %shift
}

; Test 64-bit logical shift right.
define i64 @f4(i64 %a, i64 %sh) {
; CHECK-LABEL: f4:
; CHECK: nill %r3, 31
; CHECK: srlg %r2, %r2, 0(%r3)
  %and = and i64 %sh, 31
  %shift = lshr i64 %a, %and
  ret i64 %shift
}

; Test 64-bit arithmetic shift right.
define i64 @f5(i64 %a, i64 %sh) {
; CHECK-LABEL: f5:
; CHECK: nill %r3, 31
; CHECK: srag %r2, %r2, 0(%r3)
  %and = and i64 %sh, 31
  %shift = ashr i64 %a, %and
  ret i64 %shift
}

; Test 64-bit shift left.
define i64 @f6(i64 %a, i64 %sh) {
; CHECK-LABEL: f6:
; CHECK: nill %r3, 31
; CHECK: sllg %r2, %r2, 0(%r3)
  %and = and i64 %sh, 31
  %shift = shl i64 %a, %and
  ret i64 %shift
}
