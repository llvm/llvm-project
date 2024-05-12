; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @exp2_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Exp2 %[[#]]
  %elt.exp2 = call float @llvm.exp2.f32(float %a)
  ret float %elt.exp2
}

define noundef half @exp2_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Exp2 %[[#]]
  %elt.exp2 = call half @llvm.exp2.f16(half %a)
  ret half %elt.exp2
}

declare half @llvm.exp2.f16(half)
declare float @llvm.exp2.f32(float)
