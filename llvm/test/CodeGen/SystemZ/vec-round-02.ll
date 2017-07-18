; Test v4f32 rounding on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare float @llvm.rint.f32(float)
declare float @llvm.nearbyint.f32(float)
declare float @llvm.floor.f32(float)
declare float @llvm.ceil.f32(float)
declare float @llvm.trunc.f32(float)
declare float @llvm.round.f32(float)
declare <4 x float> @llvm.rint.v4f32(<4 x float>)
declare <4 x float> @llvm.nearbyint.v4f32(<4 x float>)
declare <4 x float> @llvm.floor.v4f32(<4 x float>)
declare <4 x float> @llvm.ceil.v4f32(<4 x float>)
declare <4 x float> @llvm.trunc.v4f32(<4 x float>)
declare <4 x float> @llvm.round.v4f32(<4 x float>)

define <4 x float> @f1(<4 x float> %val) {
; CHECK-LABEL: f1:
; CHECK: vfisb %v24, %v24, 0, 0
; CHECK: br %r14
  %res = call <4 x float> @llvm.rint.v4f32(<4 x float> %val)
  ret <4 x float> %res
}

define <4 x float> @f2(<4 x float> %val) {
; CHECK-LABEL: f2:
; CHECK: vfisb %v24, %v24, 4, 0
; CHECK: br %r14
  %res = call <4 x float> @llvm.nearbyint.v4f32(<4 x float> %val)
  ret <4 x float> %res
}

define <4 x float> @f3(<4 x float> %val) {
; CHECK-LABEL: f3:
; CHECK: vfisb %v24, %v24, 4, 7
; CHECK: br %r14
  %res = call <4 x float> @llvm.floor.v4f32(<4 x float> %val)
  ret <4 x float> %res
}

define <4 x float> @f4(<4 x float> %val) {
; CHECK-LABEL: f4:
; CHECK: vfisb %v24, %v24, 4, 6
; CHECK: br %r14
  %res = call <4 x float> @llvm.ceil.v4f32(<4 x float> %val)
  ret <4 x float> %res
}

define <4 x float> @f5(<4 x float> %val) {
; CHECK-LABEL: f5:
; CHECK: vfisb %v24, %v24, 4, 5
; CHECK: br %r14
  %res = call <4 x float> @llvm.trunc.v4f32(<4 x float> %val)
  ret <4 x float> %res
}

define <4 x float> @f6(<4 x float> %val) {
; CHECK-LABEL: f6:
; CHECK: vfisb %v24, %v24, 4, 1
; CHECK: br %r14
  %res = call <4 x float> @llvm.round.v4f32(<4 x float> %val)
  ret <4 x float> %res
}

define float @f7(<4 x float> %val) {
; CHECK-LABEL: f7:
; CHECK: wfisb %f0, %v24, 0, 0
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.rint.f32(float %scalar)
  ret float %res
}

define float @f8(<4 x float> %val) {
; CHECK-LABEL: f8:
; CHECK: wfisb %f0, %v24, 4, 0
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.nearbyint.f32(float %scalar)
  ret float %res
}

define float @f9(<4 x float> %val) {
; CHECK-LABEL: f9:
; CHECK: wfisb %f0, %v24, 4, 7
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.floor.f32(float %scalar)
  ret float %res
}

define float @f10(<4 x float> %val) {
; CHECK-LABEL: f10:
; CHECK: wfisb %f0, %v24, 4, 6
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.ceil.f32(float %scalar)
  ret float %res
}

define float @f11(<4 x float> %val) {
; CHECK-LABEL: f11:
; CHECK: wfisb %f0, %v24, 4, 5
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.trunc.f32(float %scalar)
  ret float %res
}

define float @f12(<4 x float> %val) {
; CHECK-LABEL: f12:
; CHECK: wfisb %f0, %v24, 4, 1
; CHECK: br %r14
  %scalar = extractelement <4 x float> %val, i32 0
  %res = call float @llvm.round.f32(float %scalar)
  ret float %res
}
