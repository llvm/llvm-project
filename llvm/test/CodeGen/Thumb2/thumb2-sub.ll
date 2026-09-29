; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

; 171 = 0x000000ab
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: subs r0, #171
    %tmp = sub i32 %a, 171
    ret i32 %tmp
}

; 1179666 = 0x00120012
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: sub.w r0, r0, #1179666
    %tmp = sub i32 %a, 1179666
    ret i32 %tmp
}

; 872428544 = 0x34003400
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: sub.w r0, r0, #872428544
    %tmp = sub i32 %a, 872428544
    ret i32 %tmp
}

; 1448498774 = 0x56565656
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: sub.w r0, r0, #1448498774
    %tmp = sub i32 %a, 1448498774
    ret i32 %tmp
}

; 510 = 0x000001fe
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: sub.w r0, r0, #510
    %tmp = sub i32 %a, 510
    ret i32 %tmp
}

; Don't change this to an add.
define i32 @f6(i32 %a) {
; CHECK-LABEL: f6:
; CHECK: subs r0, #1
    %tmp = sub i32 %a, 1
    ret i32 %tmp
}
