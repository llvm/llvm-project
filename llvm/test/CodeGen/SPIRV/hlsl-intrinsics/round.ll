; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @round_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] RoundEven %[[#]]
  %elt.roundeven = call float @llvm.roundeven.f32(float %a)
  ret float %elt.roundeven
}

define noundef half @round_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] RoundEven %[[#]]
  %elt.roundeven = call half @llvm.roundeven.f16(half %a)
  ret half %elt.roundeven
}

declare half @llvm.roundeven.f16(half)
declare float @llvm.roundeven.f32(float)
