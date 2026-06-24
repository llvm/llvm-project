; RUN: llc -mtriple=aarch64 -o - %s | FileCheck %s

define i32 @add(i32 %lhs, i32 %rhs) {
entry:
  %sum = add i32 %lhs, %rhs
  ret i32 %sum
}

; CHECK-LABEL: add:
; CHECK: add w0, w0, w1
; CHECK: ret
