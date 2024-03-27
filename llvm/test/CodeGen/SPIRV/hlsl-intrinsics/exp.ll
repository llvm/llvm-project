; RUN: llc -O0 -mtriple=spirv-unknown-linux %s -o - | FileCheck %s

; CHECK: OpExtInstImport "GLSL.std.450"

define noundef float @exp_float(float noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Exp %[[#]]
  %elt.exp = call float @llvm.exp.f32(float %a)
  ret float %elt.exp
}

define noundef half @exp_half(half noundef %a) {
entry:
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] Exp %[[#]]
  %elt.exp = call half @llvm.exp.f16(half %a)
  ret half %elt.exp
}

declare half @llvm.exp.f16(half)
declare float @llvm.exp.f32(float)
