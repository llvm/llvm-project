; Test strict vector addition on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.fadd.v4f32(<4 x float>, <4 x float>, metadata, metadata)

; Test a v4f32 addition.
define <4 x float> @f1(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2) {
; CHECK-LABEL: f1:
; CHECK: vfasb %v24, %v26, %v28
; CHECK: br %r14
  %ret = call <4 x float> @llvm.experimental.constrained.fadd.v4f32(
                        <4 x float> %val1, <4 x float> %val2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %ret
}

; Test an f32 addition that uses vector registers.
define float @f2(<4 x float> %val1, <4 x float> %val2) {
; CHECK-LABEL: f2:
; CHECK: wfasb %f0, %v24, %v26
; CHECK: br %r14
  %scalar1 = extractelement <4 x float> %val1, i32 0
  %scalar2 = extractelement <4 x float> %val2, i32 0
  %ret = call float @llvm.experimental.constrained.fadd.f32(
                        float %scalar1, float %scalar2,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %ret
}
