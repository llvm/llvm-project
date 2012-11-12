; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

; 734439407618 = 0x000000ab00000002
define i64 @f1(i64 %a) {
; CHECK: f1:
; CHECK: adds r0, #2
    %tmp = add i64 %a, 734439407618
    ret i64 %tmp
}

; 5066626890203138 = 0x0012001200000002
define i64 @f2(i64 %a) {
; CHECK: f2:
; CHECK: adds r0, #2
    %tmp = add i64 %a, 5066626890203138
    ret i64 %tmp
}

; 3747052064576897026 = 0x3400340000000002
define i64 @f3(i64 %a) {
; CHECK: f3:
; CHECK: adds r0, #2
    %tmp = add i64 %a, 3747052064576897026
    ret i64 %tmp
}

; 6221254862626095106 = 0x5656565600000002
define i64 @f4(i64 %a) {
; CHECK: f4:
; CHECK: adds r0, #2
    %tmp = add i64 %a, 6221254862626095106 
    ret i64 %tmp
}

; 287104476244869122 = 0x03fc000000000002
define i64 @f5(i64 %a) {
; CHECK: f5:
; CHECK: adds r0, #2
    %tmp = add i64 %a, 287104476244869122
    ret i64 %tmp
}

define i64 @f6(i64 %a, i64 %b) {
; CHECK: f6:
; CHECK: adds r0, r0, r2
    %tmp = add i64 %a, %b
    ret i64 %tmp
}
