; RUN: llc -march=xcore < %s | FileCheck %s

define i32 @f(i1 %a) {
entry:
; CHECK: f
; CHECK: zext r0, 1
; CHECK: retsp 0
  %b= zext i1 %a to i32
  ret i32 %b
}
