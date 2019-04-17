; RUN: opt -licm -S < %s | FileCheck %s

define void @f() {
; CHECK-LABEL: @f(
entry:
  br label %bb0

bb0:
  %tobool7 = icmp eq i1 undef, undef
  br label %bb1

bb1:
  br i1 undef, label %bb0, label %bb0

unreachable:
; CHECK-LABEL: unreachable:
; CHECK:   br i1 undef, label %unreachable, label %unreachable
  br i1 %tobool7, label %unreachable, label %unreachable

bb3:
  unreachable
}
