; Test strict v4f32 rounding on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.experimental.constrained.rint.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.nearbyint.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.floor.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.ceil.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.trunc.f32(float, metadata, metadata)
declare float @llvm.experimental.constrained.round.f32(float, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.rint.v4f32(<4 x float>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.nearbyint.v4f32(<4 x float>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.floor.v4f32(<4 x float>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.ceil.v4f32(<4 x float>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.trunc.v4f32(<4 x float>, metadata, metadata)
declare <4 x float> @llvm.experimental.constrained.round.v4f32(<4 x float>, metadata, metadata)

define <4 x float> @f1(<4 x float> %val) {
; CHECK-LABEL: f1:
; CHECK: vfisb %v24, %v24, 0, 0
; CHECK: br %r14
  %res = call <4 x float> @llvm.experimental.constrained.rint.v4f32(
                        <4 x float> %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %res
}

define <4 x float> @f2(<4 x float> %val) {
; CHECK-LABEL: f2:
; CHECK: vfisb %v24, %v24, 4, 0
; CHECK: br %r14
  %res = call <4 x float> @llvm.experimental.constrained.nearbyint.v4f32(
                        <4 x float> %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %res
}

define <4 x float> @f3(<4 x float> %val) {
; CHECK-LABEL: f3:
; CHECK: vfisb %v24, %v24, 4, 7
; CHECK: br %r14
  %res = call <4 x float> @llvm.experimental.constrained.floor.v4f32(
                        <4 x float> %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %res
}

define <4 x float> @f4(<4 x float> %val) {
; CHECK-LABEL: f4:
; CHECK: vfisb %v24, %v24, 4, 6
; CHECK: br %r14
  %res = call <4 x float> @llvm.experimental.constrained.ceil.v4f32(
                        <4 x float> %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %res
}

define <4 x float> @f5(<4 x float> %val) {
; CHECK-LABEL: f5:
; CHECK: vfisb %v24, %v24, 4, 5
; CHECK: br %r14
  %res = call <4 x float> @llvm.experimental.constrained.trunc.v4f32(
                        <4 x float> %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %res
}

define <4 x float> @f6(<4 x float> %val) {
; CHECK-LABEL: f6:
; CHECK: vfisb %v24, %v24, 4, 1
; CHECK: br %r14
  %res = call <4 x float> @llvm.experimental.constrained.round.v4f32(
                        <4 x float> %val,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret <4 x float> %res
}

define float @f7(<4 x float> %val) {
; CHECK-LABEL: f7:
; CHECK: wfisb %f0, %v24, 0, 0
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.experimental.constrained.rint.f32(
                        float %scalar,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

define float @f8(<4 x float> %val) {
; CHECK-LABEL: f8:
; CHECK: wfisb %f0, %v24, 4, 0
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.experimental.constrained.nearbyint.f32(
                        float %scalar,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

define float @f9(<4 x float> %val) {
; CHECK-LABEL: f9:
; CHECK: wfisb %f0, %v24, 4, 7
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.experimental.constrained.floor.f32(
                        float %scalar,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

define float @f10(<4 x float> %val) {
; CHECK-LABEL: f10:
; CHECK: wfisb %f0, %v24, 4, 6
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.experimental.constrained.ceil.f32(
                        float %scalar,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

define float @f11(<4 x float> %val) {
; CHECK-LABEL: f11:
; CHECK: wfisb %f0, %v24, 4, 5
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.experimental.constrained.trunc.f32(
                        float %scalar,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}

define float @f12(<4 x float> %val) {
; CHECK-LABEL: f12:
; CHECK: wfisb %f0, %v24, 4, 1
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.experimental.constrained.round.f32(
                        float %scalar,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict")
  ret float %res
}
