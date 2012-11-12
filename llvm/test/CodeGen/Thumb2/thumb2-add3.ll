; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i32 @f1(i32 %a) {
    %tmp = add i32 %a, 4095
    ret i32 %tmp
}

; CHECK: f1:
; CHECK: 	addw	r0, r0, #4095
