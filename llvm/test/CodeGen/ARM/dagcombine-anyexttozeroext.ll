; RUN: llc -mtriple armv7 %s -o - | FileCheck %s

; CHECK: f:
define float @f(<4 x i16>* nocapture %in) {
  ; CHECK: vldr
  ; CHECK: vmovl.u16
  ; CHECK-NOT: vand
  %1 = load <4 x i16>* %in
  ; CHECK: vcvt.f32.u32
  %2 = uitofp <4 x i16> %1 to <4 x float>
  %3 = extractelement <4 x float> %2, i32 0
  %4 = extractelement <4 x float> %2, i32 1
  %5 = extractelement <4 x float> %2, i32 2

  ; CHECK: vadd.f32
  %6 = fadd float %3, %4
  %7 = fadd float %6, %5

  ret float %7
}

define float @g(<4 x i16>* nocapture %in) {
  ; CHECK: vldr
  %1 = load <4 x i16>* %in
  ; CHECK-NOT: uxth
  %2 = extractelement <4 x i16> %1, i32 0
  ; CHECK: vcvt.f32.u32
  %3 = uitofp i16 %2 to float
  ret float %3
}
