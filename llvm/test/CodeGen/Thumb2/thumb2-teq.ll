; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

; These tests would be improved by 'movs r0, #0' being rematerialized below the
; test as 'mov.w r0, #0'.

; 0x000000bb = 187
define i1 @f2(i32 %a) {
    %tmp = xor i32 %a, 187
    %tmp1 = icmp eq i32 0, %tmp
    ret i1 %tmp1
}
; CHECK-LABEL: f2:
; CHECK: 	teq.w	{{.*}}, #187

; 0x00aa00aa = 11141290
define i1 @f3(i32 %a) {
    %tmp = xor i32 %a, 11141290 
    %tmp1 = icmp eq i32 %tmp, 0
    ret i1 %tmp1
}
; CHECK-LABEL: f3:
; CHECK: 	teq.w	{{.*}}, #11141290

; 0xcc00cc00 = 3422604288
define i1 @f6(i32 %a) {
    %tmp = xor i32 %a, 3422604288
    %tmp1 = icmp eq i32 0, %tmp
    ret i1 %tmp1
}
; CHECK-LABEL: f6:
; CHECK: 	teq.w	{{.*}}, #-872363008

; 0xdddddddd = 3722304989
define i1 @f7(i32 %a) {
    %tmp = xor i32 %a, 3722304989
    %tmp1 = icmp eq i32 %tmp, 0
    ret i1 %tmp1
}
; CHECK-LABEL: f7:
; CHECK: 	teq.w	{{.*}}, #-572662307

; 0xdddddddd = 3722304989
define i1 @f8(i32 %a) {
    %tmp = xor i32 %a, 3722304989
    %tmp1 = icmp ne i32 0, %tmp
    ret i1 %tmp1
}

; 0x00110000 = 1114112
define i1 @f10(i32 %a) {
    %tmp = xor i32 %a, 1114112
    %tmp1 = icmp eq i32 0, %tmp
    ret i1 %tmp1
}
; CHECK-LABEL: f10:
; CHECK: 	teq.w	{{.*}}, #1114112

