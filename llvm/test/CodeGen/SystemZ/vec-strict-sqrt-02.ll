; Test strict f32 and v4f32 square root on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.experimental.constrained.sqrt.f32(float, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.sqrt.v4f32(<4 x float>, metadata, metadata)

define <4 x float> @f1(<4 x float> %val) {
; CHECK-LABEL: f1:
; CHECK: vfsqsb %v24, %v24
; CHECK: br %r14
  %ret = call <4 x float> @llvm.experimental.constrained.sqrt.v4f32(
                        <4 x float> %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %ret
}

define float @f2(<4 x float> %val) {
; CHECK-LABEL: f2:
; CHECK: wfsqsb %f0, %v24
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %ret = call float @llvm.experimental.constrained.sqrt.f32(
                        float %scalar,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %ret
}
