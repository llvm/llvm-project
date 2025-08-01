; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: %{{.*}} = G_INTRINSIC intrinsic(@llvm.spv.refract), %{{.*}}, %{{.*}}, %{{.*}} is only supported with the GLSL extended instruction set.

define noundef <4 x float> @refract_float4(<4 x float> noundef %I, <4 x float> noundef %N, float noundef %ETA) {
entry:
  %spv.refract = call <4 x float> @llvm.spv.refract.f32(<4 x float> %I, <4 x float> %N, float %ETA)
  ret <4 x float> %spv.refract
}

declare <4 x float> @llvm.spv.refract.f32(<4 x float>, <4 x float>, float)
