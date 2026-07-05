; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2,+32bit %s -o - \
; RUN:  | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
; CHECK-LABEL: f1:
; CHECK: subs.w r0, r0, r2
; To test dead_carry, +32bit prevents sbc conveting to 16-bit sbcs
; CHECK: sbc.w  r1, r1, r3
    %tmp = sub i64 %a, %b
    ret i64 %tmp
}
