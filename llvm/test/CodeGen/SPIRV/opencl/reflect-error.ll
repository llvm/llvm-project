; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 | FileCheck %s
; RUN: not %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o /dev/null 2>&1 -filetype=obj %}
; RUN: not %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o /dev/null 2>&1 -filetype=obj %}

; CHECK: LLVM ERROR: Intrinsic selection not supported for this instruction set: %{{.*}} = G_INTRINSIC intrinsic(@llvm.spv.reflect), %{{.*}}, %{{.*}}

define noundef <4 x half> @reflect_half4(<4 x half> noundef %a, <4 x half> noundef %b) {
entry:
  %spv.reflect = call <4 x half> @llvm.spv.reflect.f16(<4 x half> %a, <4 x half> %b)
  ret <4 x half> %spv.reflect
}

define noundef <4 x float> @reflect_float4(<4 x float> noundef %a, <4 x float> noundef %b) {
entry:
  %spv.reflect = call <4 x float> @llvm.spv.reflect.f32(<4 x float> %a, <4 x float> %b)
  ret <4 x float> %spv.reflect
}

declare <4 x half> @llvm.spv.reflect.f16(<4 x half>, <4 x half>)
declare <4 x float> @llvm.spv.reflect.f32(<4 x float>, <4 x float>)

