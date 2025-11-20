; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: %{{.*}} = G_INTRINSIC intrinsic(@llvm.spv.reflect), %{{.*}}, %{{.*}} is only supported with the GLSL extended instruction set.

define noundef <4 x float> @reflect_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
  %spv.reflect = call <4 x float> @llvm.spv.reflect.f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %spv.reflect
}

declare <4 x float> @llvm.spv.reflect.f32(<4 x float>, <4 x float>)

