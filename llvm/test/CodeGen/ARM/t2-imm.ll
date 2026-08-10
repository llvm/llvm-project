; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

define i32 @f6(i32 %a) {
; CHECK:f6
; CHECK: movw r0, #1123
; CHECK: movt r0, #1000
    %tmp = add i32 0, 65537123
    ret i32 %tmp
}
