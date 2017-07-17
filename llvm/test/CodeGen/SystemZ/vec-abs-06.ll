; Test f32 and v4f32 absolute on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.fabs.f32(float)
declare <4 x float> @llvm.fabs.v4f32(<4 x float>)

; Test a plain absolute.
define <4 x float> @f1(<4 x float> %val) {
; CHECK-LABEL: f1:
; CHECK: vflpsb %v24, %v24
; CHECK: br %r14
  %ret = call <4 x float> @llvm.fabs.v4f32(<4 x float> %val)
  ret <4 x float> %ret
}

; Test a negative absolute.
define <4 x float> @f2(<4 x float> %val) {
; CHECK-LABEL: f2:
; CHECK: vflnsb %v24, %v24
; CHECK: br %r14
  %abs = call <4 x float> @llvm.fabs.v4f32(<4 x float> %val)
  %ret = fsub <4 x float> <float -0.0, float -0.0,
                           float -0.0, float -0.0>, %abs
  ret <4 x float> %ret
}

; Test an f32 absolute that uses vector registers.
define float @f3(<4 x float> %val) {
; CHECK-LABEL: f3:
; CHECK: wflpsb %f0, %v24
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %ret = call float @llvm.fabs.f32(float %scalar)
  ret float %ret
}

; Test an f32 negative absolute that uses vector registers.
define float @f4(<4 x float> %val) {
; CHECK-LABEL: f4:
; CHECK: wflnsb %f0, %v24
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %abs = call float @llvm.fabs.f32(float %scalar)
  %ret = fsub float -0.0, %abs
  ret float %ret
}
