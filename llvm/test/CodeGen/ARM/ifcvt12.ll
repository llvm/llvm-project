; RUN: llc < %s -mtriple=arm-apple-darwin -mcpu=cortex-a8 | FileCheck %s
define i32 @f1(i32 %a, i32 %b, i32 %c) {
; CHECK-LABEL: f1:
; CHECK: mlsne r0, r0, r1, r2
    %tmp1 = icmp eq i32 %a, 0
    br i1 %tmp1, label %cond_false, label %cond_true

cond_true:
    %tmp2 = mul i32 %a, %b
    %tmp3 = sub i32 %c, %tmp2
    ret i32 %tmp3

cond_false:
    ret i32 %a
}
