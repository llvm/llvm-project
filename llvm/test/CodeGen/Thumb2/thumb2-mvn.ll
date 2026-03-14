; RUN: llc -mtriple=thumbv7-apple-darwin %s -o - | FileCheck %s

; 0x000000bb = 187
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: mvn r0, #187
    %tmp = xor i32 4294967295, 187
    ret i32 %tmp
}

; 0x00aa00aa = 11141290
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: mvn r0, #11141290
    %tmp = xor i32 4294967295, 11141290 
    ret i32 %tmp
}

; 0xcc00cc00 = 3422604288
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: mvn r0, #-872363008
    %tmp = xor i32 4294967295, 3422604288
    ret i32 %tmp
}

; 0x00110000 = 1114112
define i32 @f5(i32 %a) {
; CHECK-LABEL: f5:
; CHECK: mvn r0, #1114112
    %tmp = xor i32 4294967295, 1114112
    ret i32 %tmp
}
