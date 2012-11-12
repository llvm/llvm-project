; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s

define i64 @f1(i64 %a, i64 %b) {
; CHECK: f1:
; CHECK: adds r0, r0, r2
; CHECK: adcs r1, r3
    %tmp = add i64 %a, %b
    ret i64 %tmp
}
