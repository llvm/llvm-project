; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: asrs r0, r0, #17
    %tmp = ashr i32 %a, 17
    ret i32 %tmp
}
