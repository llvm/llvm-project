; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @trunc_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Trunc %[[#]]
  %elt.trunc = call float @llvm.trunc.f32(float %a)
  ret float %elt.trunc
}

define noundef half @trunc_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Trunc %[[#]]
  %elt.trunc = call half @llvm.trunc.f16(half %a)
  ret half %elt.trunc
}

declare half @llvm.trunc.f16(half)
declare float @llvm.trunc.f32(float)
