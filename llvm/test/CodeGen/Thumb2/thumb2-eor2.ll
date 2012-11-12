; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; 0x000000bb = 187
define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: eor {{.*}}#187
    %tmp = xor i32 %a, 187
    ret i32 %tmp
}

; 0x00aa00aa = 11141290
define i32 @f2(i32 %a) {
; CHECK: f2:
; CHECK: eor {{.*}}#11141290
    %tmp = xor i32 %a, 11141290 
    ret i32 %tmp
}

; 0xcc00cc00 = 3422604288
define i32 @f3(i32 %a) {
; CHECK: f3:
; CHECK: eor {{.*}}#-872363008
    %tmp = xor i32 %a, 3422604288
    ret i32 %tmp
}

; 0xdddddddd = 3722304989
define i32 @f4(i32 %a) {
; CHECK: f4:
; CHECK: eor {{.*}}#-572662307
    %tmp = xor i32 %a, 3722304989
    ret i32 %tmp
}

; 0x00110000 = 1114112
define i32 @f5(i32 %a) {
; CHECK: f5:
; CHECK: eor {{.*}}#1114112
    %tmp = xor i32 %a, 1114112
    ret i32 %tmp
}
