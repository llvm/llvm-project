; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: %{{.*}} = G_INTRINSIC intrinsic(@llvm.spv.faceforward), %{{.*}}, %{{.*}}, %{{.*}} is only supported with the GLSL extended instruction set.

define noundef <4 x float> @faceforward_float4(<4 x float> noundef %a, <4 x float> noundef %b, <4 x float> noundef %c) {
entry:
  %spv.faceforward = call <4 x float> @llvm.spv.faceforward.v4f32(<4 x float> %a, <4 x float> %b, <4 x float> %c)
  ret <4 x float> %spv.faceforward
}

declare <4 x float> @llvm.spv.faceforward.v4f32(<4 x float>, <4 x float>, <4 x float>)

