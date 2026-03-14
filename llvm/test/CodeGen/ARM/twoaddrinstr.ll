; Tests for the two-address instruction pass.
; RUN: llc -mtriple=arm-eabi -mcpu=cortex-a9 -arm-atomic-cfg-tidy=0 %s -o - | FileCheck %s

define void @PR13378() nounwind {
; This was orriginally a crasher trying to schedule the instructions.
; CHECK-LABEL:      PR13378:
; CHECK:        vmov.i32
; CHECK-NEXT:   vld1.32
; CHECK-NEXT:   vst1.32
; CHECK-NEXT:   vst1.32
; CHECK-NEXT:   vmov.f32
; CHECK-NEXT:   vmov.f32
; CHECK-NEXT:   vst1.32

entry:
  %0 = load <4 x float>, ptr undef, align 4
  store <4 x float> zeroinitializer, ptr undef, align 4
  store <4 x float> %0, ptr undef, align 4
  %1 = insertelement <4 x float> %0, float 1.000000e+00, i32 3
  store <4 x float> %1, ptr undef, align 4
  unreachable
}
