; RUN: llc < %s -mtriple=armv7-eabi -mcpu=cortex-a8
; PR5055

module asm ".globl\09__aeabi_f2lz"
module asm ".set\09__aeabi_f2lz, __fixsfdi"
module asm ""

define arm_aapcs_vfpcc i64 @__fixsfdi(float %a) nounwind {
entry:
  %0 = fcmp olt float %a, 0.000000e+00            ; <i1> [#uses=1]
  br i1 %0, label %bb, label %bb1

bb:                                               ; preds = %entry
  %1 = fsub float -0.000000e+00, %a               ; <float> [#uses=1]
  %2 = tail call arm_aapcs_vfpcc  i64 @__fixunssfdi(float %1) nounwind ; <i64> [#uses=1]
  %3 = sub i64 0, %2                              ; <i64> [#uses=1]
  ret i64 %3

bb1:                                              ; preds = %entry
  %4 = tail call arm_aapcs_vfpcc  i64 @__fixunssfdi(float %a) nounwind ; <i64> [#uses=1]
  ret i64 %4
}

declare arm_aapcs_vfpcc i64 @__fixunssfdi(float)
