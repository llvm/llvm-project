; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: lsls r0, r0, #5
    %tmp = shl i32 %a, 5
    ret i32 %tmp
}
