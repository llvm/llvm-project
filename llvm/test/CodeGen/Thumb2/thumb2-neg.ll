; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: rsbs r0, r0, #0
    %tmp = sub i32 0, %a
    ret i32 %tmp
}
