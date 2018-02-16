; RUN: llc -mtriple=thumb-eabi -mcpu=arm1156t2-s -mattr=+thumb2 %s -o - | FileCheck %s

; -0x000000bb = 4294967109
define i32 @f1(i32 %a) {
; CHECK-LABEL: f1:
; CHECK: adds {{r.*}}, #187
    %tmp = icmp ne i32 %a, 4294967109
    %ret = select i1 %tmp, i32 42, i32 24
    ret i32 %ret
}

; -0x00aa00aa = 4283826006
define i32 @f2(i32 %a) {
; CHECK-LABEL: f2:
; CHECK: cmn.w {{r.*}}, #11141290
    %tmp = icmp eq i32 %a, 4283826006
    %ret = select i1 %tmp, i32 42, i32 24
    ret i32 %ret
}

; -0xcc00cc00 = 872363008
define i32 @f3(i32 %a) {
; CHECK-LABEL: f3:
; CHECK: cmn.w {{r.*}}, #-872363008
    %tmp = icmp ne i32 %a, 872363008
    %ret = select i1 %tmp, i32 42, i32 24
    ret i32 %ret
}

; -0x00110000 = 4293853184
define i32 @f4(i32 %a) {
; CHECK-LABEL: f4:
; CHECK: cmn.w {{r.*}}, #1114112
    %tmp = icmp eq i32 %a, 4293853184
    %ret = select i1 %tmp, i32 42, i32 24
    ret i32 %ret
}
