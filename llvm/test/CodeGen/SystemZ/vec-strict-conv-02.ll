; Test conversions between different-sized float elements.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

declare <2 x float> @llvm.experimental.constrained.fptrunc.v2f32.v2f64(<2 x double>, metadata, metadata)
declare float @llvm.experimental.constrained.fptrunc.f32.f64(double, metadata, metadata)

declare <2 x double> @llvm.experimental.constrained.fpext.v2f64.v2f32(<2 x float>, metadata)
declare double @llvm.experimental.constrained.fpext.f64.f32(float, metadata)

; Test cases where both elements of a v2f64 are converted to f32s.
define void @f1(<2 x double> %val, <2 x float> *%ptr) {
; CHECK-LABEL: f1:
; CHECK: vledb {{%v[0-9]+}}, %v24, 0, 0
; CHECK: br %r14
  %res = call <2 x float> @llvm.experimental.constrained.fptrunc.v2f32.v2f64(
                                               <2 x double> %val,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  store <2 x float> %res, <2 x float> *%ptr
  ret void
}

; Test conversion of an f64 in a vector register to an f32.
define float @f2(<2 x double> %vec) #0 {
; CHECK-LABEL: f2:
; CHECK: wledb %f0, %v24, 0, 0
; CHECK: br %r14
  %scalar = extractelement <2 x double> %vec, i32 0
  %ret = call float @llvm.experimental.constrained.fptrunc.f32.f64(
                                               double %scalar,
                                               metadata !"round.dynamic",
                                               metadata !"fpexcept.strict") #0
  ret float %ret
}

; Test cases where even elements of a v4f32 are converted to f64s.
define <2 x double> @f3(<4 x float> %vec) {
; CHECK-LABEL: f3:
; CHECK: vldeb %v24, {{%v[0-9]+}}
; CHECK: br %r14
  %shuffle = shufflevector <4 x float> %vec, <4 x float> undef, <2 x i32> <i32 0, i32 2>
  %res = call <2 x double> @llvm.experimental.constrained.fpext.v2f64.v2f32(
                                               <2 x float> %shuffle,
                                               metadata !"fpexcept.strict") #0
  ret <2 x double> %res
}

; Test conversion of an f32 in a vector register to an f64.
define double @f4(<4 x float> %vec) {
; CHECK-LABEL: f4:
; CHECK: wldeb %f0, %v24
; CHECK: br %r14
  %scalar = extractelement <4 x float> %vec, i32 0
  %ret = call double @llvm.experimental.constrained.fpext.f64.f32(
                                               float %scalar,
                                               metadata !"fpexcept.strict") #0
  ret double %ret
}

attributes #0 = { strictfp }
