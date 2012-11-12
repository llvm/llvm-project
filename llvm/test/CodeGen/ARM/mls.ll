; RUN: llc < %s -march=arm -mattr=+v6t2 | FileCheck %s
; RUN: llc < %s -march=arm -mattr=+v6t2 -arm-use-mulops=false | FileCheck %s -check-prefix=NO_MULOPS

define i32 @f1(i32 %a, i32 %b, i32 %c) {
    %tmp1 = mul i32 %a, %b
    %tmp2 = sub i32 %c, %tmp1
    ret i32 %tmp2
}

; sub doesn't commute, so no mls for this one
define i32 @f2(i32 %a, i32 %b, i32 %c) {
    %tmp1 = mul i32 %a, %b
    %tmp2 = sub i32 %tmp1, %c
    ret i32 %tmp2
}

; CHECK: f1:
; CHECK: mls	r0, r0, r1, r2
; NO_MULOPS: f1:
; NO_MULOPS: mul r0, r0, r1
; NO_MULOPS-NEXT: sub r0, r2, r0

; CHECK: f2:
; CHECK: mul r0, r0, r1
; CHECK-NEXT: sub r0, r0, r2
; NO_MULOPS: f2:
; NO_MULOPS: mul r0, r0, r1
; NO_MULOPS-NEXT: sub r0, r0, r2
