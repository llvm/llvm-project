; RUN: opt -passes=instsimplify -S %s | FileCheck %s

; The false edge of (x == 0 || other) proves x is nonzero.
define i1 @or_false(i64 %x, i1 %other) {
; CHECK-LABEL: define i1 @or_false(
; CHECK: known:
; CHECK-NEXT: ret i1 false
entry:
  %iszero = icmp eq i64 %x, 0
  %either = or i1 %iszero, %other
  br i1 %either, label %maybe, label %known

known:
  %neg = sub i64 0, %x
  %result = icmp eq i64 %neg, 0
  ret i1 %result

maybe:
  ret i1 false
}

; A select can represent the same logical OR.
define i1 @select_false(i64 %x, i1 %other) {
; CHECK-LABEL: define i1 @select_false(
; CHECK: known:
; CHECK-NEXT: ret i1 false
entry:
  %iszero = icmp eq i64 %x, 0
  %either = select i1 %iszero, i1 true, i1 %other
  br i1 %either, label %maybe, label %known

known:
  %neg = sub i64 0, %x
  %result = icmp eq i64 %neg, 0
  ret i1 %result

maybe:
  ret i1 false
}

; The true edge of the OR does not establish that x is nonzero.
define i1 @or_true(i64 %x, i1 %other) {
; CHECK-LABEL: define i1 @or_true(
; CHECK: maybe:
; CHECK: %result = icmp eq i64 %neg, 0
entry:
  %iszero = icmp eq i64 %x, 0
  %either = or i1 %iszero, %other
  br i1 %either, label %maybe, label %known

known:
  ret i1 false

maybe:
  %neg = sub i64 0, %x
  %result = icmp eq i64 %neg, 0
  ret i1 %result
}
