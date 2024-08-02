; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @sin_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Sin %[[#]]
  %elt.sin = call float @llvm.sin.f32(float %a)
  ret float %elt.sin
}

define noundef half @sin_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Sin %[[#]]
  %elt.sin = call half @llvm.sin.f16(half %a)
  ret half %elt.sin
}

declare half @llvm.sin.f16(half)
declare float @llvm.sin.f32(float)
