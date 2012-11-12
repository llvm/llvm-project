; RUN: llc < %s -march=arm -mattr=+v6t2 | FileCheck %s

; 4278190095 = 0xff00000f
define i32 @f1(i32 %a) {
; CHECK: f1:
; CHECK: bfc
    %tmp = and i32 %a, 4278190095
    ret i32 %tmp
}

; 4286578688 = 0xff800000
define i32 @f2(i32 %a) {
; CHECK: f2:
; CHECK: bfc
    %tmp = and i32 %a, 4286578688
    ret i32 %tmp
}

; 4095 = 0x00000fff
define i32 @f3(i32 %a) {
; CHECK: f3:
; CHECK: bfc
    %tmp = and i32 %a, 4095
    ret i32 %tmp
}
