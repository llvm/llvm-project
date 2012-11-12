; RUN: llc -mtriple armv7 %s -o - | FileCheck %s

; CHECK: f:
define float @f(<4 x i16>* nocapture %in) {
  ; CHECK: vldr
  ; CHECK: vmovl.u16
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

; CHECK: g:
define float @g(<4 x i8>* nocapture %in) {
; Note: vld1 here is reasonably important. Mixing VFP and NEON
; instructions is bad on some cores
  ; CHECK: vld1
  ; CHECK: vmovl.u8
  ; CHECK: vmovl.u16
  %1 = load <4 x i8>* %in
  ; CHECK: vcvt.f32.u32
  %2 = uitofp <4 x i8> %1 to <4 x float>
  %3 = extractelement <4 x float> %2, i32 0
  %4 = extractelement <4 x float> %2, i32 1
  %5 = extractelement <4 x float> %2, i32 2

  ; CHECK: vadd.f32
  %6 = fadd float %3, %4
  %7 = fadd float %6, %5

  ret float %7
}

; CHECK: h:
define <4 x i8> @h(<4 x float> %v) {
  ; CHECK: vcvt.{{[us]}}32.f32
  ; CHECK: vmovn.i32
  %1 = fptoui <4 x float> %v to <4 x i8>
  ret <4 x i8> %1
}

; CHECK: i:
define <4 x i8> @i(<4 x i8>* %x) {
; Note: vld1 here is reasonably important. Mixing VFP and NEON
; instructions is bad on some cores
  ; CHECK: vld1
  ; CHECK: vmovl.s8
  ; CHECK: vmovl.s16
  ; CHECK: vrecpe
  ; CHECK: vrecps
  ; CHECK: vmul
  ; CHECK: vmovn
  %1 = load <4 x i8>* %x, align 4
  %2 = sdiv <4 x i8> zeroinitializer, %1
  ret <4 x i8> %2
}
; CHECK: j:
define <4 x i32> @j(<4 x i8>* %in) nounwind {
  ; CHECK: vld1
  ; CHECK: vmovl.u8
  ; CHECK: vmovl.u16
  ; CHECK-NOT: vand
  %1 = load <4 x i8>* %in, align 4
  %2 = zext <4 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
}

