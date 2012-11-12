; RUN: llc < %s -mtriple=arm-apple-darwin | grep moveq 
; RUN: llc < %s -mtriple=armv7-apple-darwin -mcpu=cortex-a8 | FileCheck %s

define i32 @f7(float %a, float %b) {
entry:
; CHECK: f7:
; CHECK: vcmpe.f32
; CHECK: vmrs APSR_nzcv, fpscr
; CHECK: movweq
; CHECK-NOT: vmrs
; CHECK: movwvs
    %tmp = fcmp ueq float %a,%b
    %retval = select i1 %tmp, i32 666, i32 42
    ret i32 %retval
}

