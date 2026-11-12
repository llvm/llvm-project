; RUN: llc < %s -mtriple=xcore | FileCheck %s

define i32 @f(i32) nounwind {
; CHECK-LABEL: f:
; CHECK: mkmsk r0, r0
; CHECK-NEXT: retsp 0
entry:
  %1 = shl i32 1, %0
  %2 = add i32 %1, -1
  ret i32 %2
}
