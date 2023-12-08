; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @f1(i32 %a) {
    %tmp = add i32 %a, 4095
    ret i32 %tmp
}

; CHECK-LABEL: f1:
; CHECK: 	addw	r0, r0, #4095
