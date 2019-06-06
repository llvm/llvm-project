; Test strict vector subtraction on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.fsub.v4f32(<4 x float>, <4 x float>, metadata, metadata)

; Test a v4f32 subtraction.
define <4 x float> @f6(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2) {
; CHECK-LABEL: f6:
; CHECK: vfssb %v24, %v26, %v28
; CHECK: br %r14
  %ret = call <4 x float> @llvm.experimental.constrained.fsub.v4f32(
                        <4 x float> %val1, <4 x float> %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %ret
}

; Test an f32 subtraction that uses vector registers.
define float @f7(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f7:
; CHECK: wfssb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <4 x float> %val1, i32 0
  %scalar2 = extractelement <4 x float> %val2, i32 0
  %ret = call float @llvm.experimental.constrained.fsub.f32(
                        float %scalar1, float %scalar2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %ret
}
