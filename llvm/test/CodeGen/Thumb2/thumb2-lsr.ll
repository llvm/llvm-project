; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: lsrs r0, r0, #13
    %tmp = lshr i32 %a, 13
    ret i32 %tmp
}
