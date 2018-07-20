; RUN: llc -mtriple armv7 %s -o - | FileCheck %s

; CHECK-LABEL: f:
define float @f(<4 x i16>* nocapture %in) {
  ; CHECK: vld1
  ; CHECK: vmovl.u16
  ; CHECK-NOT: vand
  %1 = load <4 x i16>, <4 x i16>* %in
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

; CHECK-LABEL: g:
define float @g(<4 x i16>* nocapture %in) {
  ; CHECK: vldr
  %1 = load <4 x i16>, <4 x i16>* %in

  ; For now we're generating a vmov.16 and a uxth instruction.
  ; The uxth is redundant, and we should be able to extend without
  ; having to generate cross-domain copies. Once we can do this
  ; we should modify the checks below.

  ; CHECK: uxth
  %2 = extractelement <4 x i16> %1, i32 0
  ; CHECK: vcvt.f32.u32
  %3 = uitofp i16 %2 to float
  ret float %3
}

; Make sure we generate zext from <4 x i8> to <4 x 32>.

; CHECK-LABEL: h:
; CHECK: vld1.32
; CHECK: vmovl.u8 q8, d16
; CHECK: vmovl.u16 q8, d16
; CHECK: vmov r0, r1, d16
; CHECK: vmov r2, r3, d17
define <4 x i32> @h(<4 x i8> *%in) {
  %1 = load <4 x i8>, <4 x i8>* %in, align 4
  %2 = extractelement <4 x i8> %1, i32 0
  %3 = zext i8 %2 to i32
  %4 = insertelement <4 x i32> undef, i32 %3, i32 0
  %5 = extractelement <4 x i8> %1, i32 1
  %6 = zext i8 %5 to i32
  %7 = insertelement <4 x i32> %4, i32 %6, i32 1
  %8 = extractelement <4 x i8> %1, i32 2
  %9 = zext i8 %8 to i32
  %10 = insertelement <4 x i32> %7, i32 %9, i32 2
  %11 = extractelement <4 x i8> %1, i32 3
  %12 = zext i8 %11 to i32
  %13 = insertelement <4 x i32> %10, i32 %12, i32 3
  ret <4 x i32> %13
}
