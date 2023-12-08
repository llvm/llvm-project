; Test conversions between different-sized float elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test cases where both elements of a v2f64 are converted to f32s.
define void @f1(<2 x double> %val, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: vledb {{%v[0-9]+}}, %v24, 0, 0
; CHECK: br %r14
  %res = fptrunc <2 x double> %val to <2 x float>
  store <2 x float> %res, ptr %ptr
  ret void
}

; Test conversion of an f64 in a vector register to an f32.
define float @f2(<2 x double> %vec) {
; CHECK-LABEL: f2:
; CHECK: wledb %f0, %v24, 0, 0
; CHECK: br %r14
  %scalar = extractelement <2 x double> %vec, i32 0
  %ret = fptrunc double %scalar to float
  ret float %ret
}

; Test cases where even elements of a v4f32 are converted to f64s.
define <2 x double> @f3(<4 x float> %vec) {
; CHECK-LABEL: f3:
; CHECK: vldeb %v24, {{%v[0-9]+}}
; CHECK: br %r14
  %shuffle = shufflevector <4 x float> %vec, <4 x float> undef, <2 x i32> <i32 0, i32 2>
  %res = fpext <2 x float> %shuffle to <2 x double>
  ret <2 x double> %res
}

; Test conversion of an f32 in a vector register to an f64.
define double @f4(<4 x float> %vec) {
; CHECK-LABEL: f4:
; CHECK: wldeb %f0, %v24
; CHECK: br %r14
  %scalar = extractelement <4 x float> %vec, i32 0
  %ret = fpext float %scalar to double
  ret double %ret
}

