; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @pow_float(float noundef %a,float noundef %b) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Pow %[[#]]
  %elt.pow = call float @llvm.pow.f32(float %a,float %b)
  ret float %elt.pow
}

define noundef half @pow_half(half noundef %a, half noundef %b) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Pow %[[#]]
  %elt.pow = call half @llvm.pow.f16(half %a, half %b)
  ret half %elt.pow
}

declare half @llvm.pow.f16(half,half)
declare float @llvm.pow.f32(float,float)
