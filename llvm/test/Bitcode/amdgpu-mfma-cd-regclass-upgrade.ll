; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: llvm-dis < %s.bc | FileCheck %s
;
; The fixture was assembled from the old six/nine-argument signatures with
; an LLVM build predating cd_regclass, not by round-tripping new IR.
; These signatures require no upgrade with the call-site attribute.

declare <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float, float, <4 x float>, i32, i32, i32)
declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32, i32, i32, i32, i32, i32)

define <4 x float> @ordinary(float %a, float %b, <4 x float> %c) {
; CHECK-LABEL: define <4 x float> @ordinary(
; CHECK: call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0)
  %d = call <4 x float> @llvm.amdgcn.mfma.f32.16x16x4f32(float %a, float %b, <4 x float> %c, i32 0, i32 0, i32 0)
  ret <4 x float> %d
}

define <4 x float> @scaled(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 %sa, i32 %sb) {
; CHECK-LABEL: define <4 x float> @scaled(
; CHECK: call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 0, i32 %sa, i32 0, i32 %sb)
  %d = call <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4.v8i32.v8i32(<8 x i32> %a, <8 x i32> %b, <4 x float> %c, i32 0, i32 0, i32 0, i32 %sa, i32 0, i32 %sb)
  ret <4 x float> %d
}
