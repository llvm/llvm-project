; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @sqrt_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Sqrt %[[#]]
  %elt.sqrt = call float @llvm.sqrt.f32(float %a)
  ret float %elt.sqrt
}

define noundef half @sqrt_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Sqrt %[[#]]
  %elt.sqrt = call half @llvm.sqrt.f16(half %a)
  ret half %elt.sqrt
}

declare half @llvm.sqrt.f16(half)
declare float @llvm.sqrt.f32(float)
