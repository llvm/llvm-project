; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; 171 = 0x000000ab
define i32 @f1(i32 %a) {
    %tmp = and i32 %a, 171
    ret i32 %tmp
}
; CHECK: f1:
; CHECK: 	and	r0, r0, #171

; 1179666 = 0x00120012
define i32 @f2(i32 %a) {
    %tmp = and i32 %a, 1179666
    ret i32 %tmp
}
; CHECK: f2:
; CHECK: 	and	r0, r0, #1179666

; 872428544 = 0x34003400
define i32 @f3(i32 %a) {
    %tmp = and i32 %a, 872428544
    ret i32 %tmp
}
; CHECK: f3:
; CHECK: 	and	r0, r0, #872428544

; 1448498774 = 0x56565656
define i32 @f4(i32 %a) {
    %tmp = and i32 %a, 1448498774
    ret i32 %tmp
}
; CHECK: f4:
; CHECK: bic r0, r0, #-1448498775

; 66846720 = 0x03fc0000
define i32 @f5(i32 %a) {
    %tmp = and i32 %a, 66846720
    ret i32 %tmp
}
; CHECK: f5:
; CHECK: 	and	r0, r0, #66846720
