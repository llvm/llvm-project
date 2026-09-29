; RUN: llc < %s -o - -filetype=asm | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"
target triple = "armv8-none--eabi"

; CHECK-LABEL: fn1:
define arm_aapcs_vfpcc float @fn1(double %a, double %b, double %c, double %d, double %e, double %f, double %g, float %h, double %i, float %j) {
  ret float %j
; CHECK: vldr    s0, [sp, #8]
}

; CHECK-LABEL: fn2:
define arm_aapcs_vfpcc float @fn2(double %a, double %b, double %c, double %d, double %e, double %f, float %h, <4 x float> %i, float %j) {
  ret float %j
; CHECK: vldr    s0, [sp, #16]
}

; CHECK-LABEL: fn3:
define arm_aapcs_vfpcc float @fn3(float %a, double %b, double %c, double %d, double %e, double %f, double %g, double %h, double %i, float %j) #0 {
  ret float %j
; CHECK: vldr    s0, [sp, #8]
}
