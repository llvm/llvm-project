; RUN: llc < %s -mcpu=cortex-a8 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios"

; The double in the constant pool is 8-byte aligned, forcing the function
; alignment.
; CHECK: .p2align 3
; CHECK: func
;
; Constant pool with 8-byte entry before 4-byte entry:
; CHECK: .p2align 3
; CHECK: LCPI
; CHECK:	.long	2370821947
; CHECK:	.long	1080815255
; CHECK: LCPI
; CHECK:	.long	0x42f6e979
define void @func(ptr nocapture %x, ptr nocapture %y) nounwind ssp {
entry:
  %0 = load float, ptr %x, align 4
  %add = fadd float %0, 0x405EDD2F20000000
  store float %add, ptr %x, align 4
  %1 = load double, ptr %y, align 4
  %add1 = fadd double %1, 2.234560e+02
  store double %add1, ptr %y, align 4
  ret void
}
