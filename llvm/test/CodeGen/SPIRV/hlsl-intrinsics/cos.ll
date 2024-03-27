; RUN: llc -O0 -mtriple=spirv-unknown-linux %s -o - | FileCheck %s

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
