; Tests for the two-address instruction pass.
; RUN: llc -march=arm -mcpu=cortex-a9 < %s | FileCheck %s

define void @PR13378() nounwind {
; This was orriginally a crasher trying to schedule the instructions.
; CHECK:      PR13378:
; CHECK:        vld1.32
; CHECK-NEXT:   vst1.32
; CHECK-NEXT:   vst1.32
; CHECK-NEXT:   vmov.f32
; CHECK-NEXT:   vmov.f32
; CHECK-NEXT:   vst1.32

entry:
  %0 = load <4 x float>* undef, align 4
  store <4 x float> zeroinitializer, <4 x float>* undef, align 4
  store <4 x float> %0, <4 x float>* undef, align 4
  %1 = insertelement <4 x float> %0, float 1.000000e+00, i32 3
  store <4 x float> %1, <4 x float>* undef, align 4
  unreachable
}
