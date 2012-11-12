; RUN: llc < %s -march=thumb -mattr=+thumb2 | FileCheck %s
; RUN: llc < %s -march=thumb -mattr=+thumb2 -arm-use-mulops=false | FileCheck %s -check-prefix=NO_MULOPS

define i32 @f1(i32 %a, i32 %b, i32 %c) {
    %tmp1 = mul i32 %a, %b
    %tmp2 = add i32 %c, %tmp1
    ret i32 %tmp2
}
; CHECK: f1:
; CHECK: 	mla	r0, r0, r1, r2
; NO_MULOPS: f1:
; NO_MULOPS: muls r0, r1, r0
; NO_MULOPS-NEXT: add r0, r2

define i32 @f2(i32 %a, i32 %b, i32 %c) {
    %tmp1 = mul i32 %a, %b
    %tmp2 = add i32 %tmp1, %c
    ret i32 %tmp2
}
; CHECK: f2:
; CHECK: 	mla	r0, r0, r1, r2
; NO_MULOPS: f2:
; NO_MULOPS: muls r0, r1, r0
; NO_MULOPS-NEXT: add r0, r2
