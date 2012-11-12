; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; 171 = 0x000000ab
define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: adds r0, #171
    %tmp = add i32 %a, 171
    ret i32 %tmp
}

; 1179666 = 0x00120012
define i32 @f2(i32 %a) {
; CHECK: f2:
; CHECK: add.w r0, r0, #1179666
    %tmp = add i32 %a, 1179666
    ret i32 %tmp
}

; 872428544 = 0x34003400
define i32 @f3(i32 %a) {
; CHECK: f3:
; CHECK: add.w r0, r0, #872428544
    %tmp = add i32 %a, 872428544
    ret i32 %tmp
}

; 1448498774 = 0x56565656
define i32 @f4(i32 %a) {
; CHECK: f4:
; CHECK: add.w r0, r0, #1448498774
    %tmp = add i32 %a, 1448498774
    ret i32 %tmp
}

; 510 = 0x000001fe
define i32 @f5(i32 %a) {
; CHECK: f5:
; CHECK: add.w r0, r0, #510
    %tmp = add i32 %a, 510
    ret i32 %tmp
}
