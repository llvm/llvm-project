; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @ceil_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Ceil %[[#]]
  %elt.ceil = call float @llvm.ceil.f32(float %a)
  ret float %elt.ceil
}

define noundef half @ceil_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Ceil %[[#]]
  %elt.ceil = call half @llvm.ceil.f16(half %a)
  ret half %elt.ceil
}

declare half @llvm.ceil.f16(half)
declare float @llvm.ceil.f32(float)
