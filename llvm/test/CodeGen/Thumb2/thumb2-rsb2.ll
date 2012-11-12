; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; 171 = 0x000000ab
define i32 @f1(i32 %a) {
    %tmp = sub i32 171, %a
    ret i32 %tmp
}
; CHECK: f1:
; CHECK: 	rsb.w	r0, r0, #171

; 1179666 = 0x00120012
define i32 @f2(i32 %a) {
    %tmp = sub i32 1179666, %a
    ret i32 %tmp
}
; CHECK: f2:
; CHECK: 	rsb.w	r0, r0, #1179666

; 872428544 = 0x34003400
define i32 @f3(i32 %a) {
    %tmp = sub i32 872428544, %a
    ret i32 %tmp
}
; CHECK: f3:
; CHECK: 	rsb.w	r0, r0, #872428544

; 1448498774 = 0x56565656
define i32 @f4(i32 %a) {
    %tmp = sub i32 1448498774, %a
    ret i32 %tmp
}
; CHECK: f4:
; CHECK: 	rsb.w	r0, r0, #1448498774

; 66846720 = 0x03fc0000
define i32 @f5(i32 %a) {
    %tmp = sub i32 66846720, %a
    ret i32 %tmp
}
; CHECK: f5:
; CHECK: 	rsb.w	r0, r0, #66846720
