; RUN: llc -mtriple armv7 %s -o - | FileCheck %s

; CHECK-LABEL: f:
define float @f(ptr nocapture %in) {
  ; CHECK: vld1
  ; CHECK: vmovl.u16
  %1 = load <4 x i16>, ptr %in
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
define float @g(ptr nocapture %in) {
; Note: vld1 here is reasonably important. Mixing VFP and NEON
; instructions is bad on some cores
  ; CHECK: vld1
  ; CHECK: vmovl.u8
  ; CHECK: vmovl.u16
  %1 = load <4 x i8>, ptr %in
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

; CHECK-LABEL: h:
define <4 x i8> @h(<4 x float> %v) {
  ; CHECK: vcvt.{{[us]}}32.f32
  ; CHECK: vmovn.i32
  %1 = fptoui <4 x float> %v to <4 x i8>
  ret <4 x i8> %1
}

; CHECK-LABEL: i:
define <4 x i8> @i(ptr %x, <4 x i8> %y) {
; Note: vld1 here is reasonably important. Mixing VFP and NEON
; instructions is bad on some cores
  ; CHECK: vld1
  ; CHECK: vmovl.s8
  ; CHECK: vmovl.s16
  ; CHECK: vrecpe
  ; CHECK: vrecps
  ; CHECK: vmul
  ; CHECK: vmovn
  %1 = load <4 x i8>, ptr %x, align 4
  %2 = sdiv <4 x i8> %y, %1
  ret <4 x i8> %2
}
; CHECK-LABEL: j:
define <4 x i32> @j(ptr %in) nounwind {
  ; CHECK: vld1
  ; CHECK: vmovl.u8
  ; CHECK: vmovl.u16
  ; CHECK-NOT: vand
  %1 = load <4 x i8>, ptr %in, align 4
  %2 = zext <4 x i8> %1 to <4 x i32>
  ret <4 x i32> %2
}

