; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s


; 0x000000bb = 187
define i32 @f1(i32 %a) {
    %tmp2 = or i32 %a, 187
    ret i32 %tmp2
}
; CHECK: f1:
; CHECK: 	orr	r0, r0, #187

; 0x00aa00aa = 11141290
define i32 @f2(i32 %a) {
    %tmp2 = or i32 %a, 11141290 
    ret i32 %tmp2
}
; CHECK: f2:
; CHECK: 	orr	r0, r0, #11141290

; 0xcc00cc00 = 3422604288
define i32 @f3(i32 %a) {
    %tmp2 = or i32 %a, 3422604288
    ret i32 %tmp2
}
; CHECK: f3:
; CHECK: 	orr	r0, r0, #-872363008

; 0x44444444 = 1145324612
define i32 @f4(i32 %a) {
    %tmp2 = or i32 %a, 1145324612
    ret i32 %tmp2
}
; CHECK: f4:
; CHECK: 	orr	r0, r0, #1145324612

; 0x00110000 = 1114112
define i32 @f5(i32 %a) {
    %tmp2 = or i32 %a, 1114112
    ret i32 %tmp2
}
; CHECK: f5:
; CHECK: 	orr	r0, r0, #1114112
