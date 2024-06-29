; RUN: llc -O0 -mtriple=spirv-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @log2_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Log2 %[[#]]
  %elt.log2 = call float @llvm.log2.f32(float %a)
  ret float %elt.log2
}

define noundef half @log2_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Log2 %[[#]]
  %elt.log2 = call half @llvm.log2.f16(half %a)
  ret half %elt.log2
}

declare half @llvm.log2.f16(half)
declare float @llvm.log2.f32(float)
