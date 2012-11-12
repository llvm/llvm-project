; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; -0x000000bb = 4294967109
define i1 @f1(i32 %a) {
; CHECK: f1:
; CHECK: cmn.w {{r.*}}, #187
    %tmp = icmp ne i32 %a, 4294967109
    ret i1 %tmp
}

; -0x00aa00aa = 4283826006
define i1 @f2(i32 %a) {
; CHECK: f2:
; CHECK: cmn.w {{r.*}}, #11141290
    %tmp = icmp eq i32 %a, 4283826006
    ret i1 %tmp
}

; -0xcc00cc00 = 872363008
define i1 @f3(i32 %a) {
; CHECK: f3:
; CHECK: cmn.w {{r.*}}, #-872363008
    %tmp = icmp ne i32 %a, 872363008
    ret i1 %tmp
}

; -0x00110000 = 4293853184
define i1 @f4(i32 %a) {
; CHECK: f4:
; CHECK: cmn.w {{r.*}}, #1114112
    %tmp = icmp eq i32 %a, 4293853184
    ret i1 %tmp
}
