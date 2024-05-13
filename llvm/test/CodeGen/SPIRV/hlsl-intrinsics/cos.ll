; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @cos_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Cos %[[#]]
  %elt.cos = call float @llvm.cos.f32(float %a)
  ret float %elt.cos
}

define noundef half @cos_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Cos %[[#]]
  %elt.cos = call half @llvm.cos.f16(half %a)
  ret half %elt.cos
}

declare half @llvm.cos.f16(half)
declare float @llvm.cos.f32(float)
