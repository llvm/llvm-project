; RUN: opt -passes=licm -S < %s | FileCheck %s

define void @f(i1 %c) {
; CHECK-LABEL: @f(
entry:
  br label %bb0

bb0:
  %tobool7 = icmp eq i1 0, 1
  br label %bb1

bb1:
  br i1 %c, label %bb0, label %bb0

unreachable:
; CHECK-LABEL: unreachable:
; CHECK:   br i1 poison, label %unreachable, label %unreachable
  br i1 %tobool7, label %unreachable, label %unreachable

bb3:
  unreachable
}
