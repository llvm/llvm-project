; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for round are generated for float and half.

; CHECK-LABEL: round_half
define noundef half @round_half(half noundef %a) {
entry:
; CHECK: call half @dx.op.unary.f16(i32 26, half %{{.*}})
  %elt.roundeven = call half @llvm.roundeven.f16(half %a)
  ret half %elt.roundeven
}

; CHECK-LABEL: round_float
define noundef float @round_float(float noundef %a) {
entry:
; CHECK: call float @dx.op.unary.f32(i32 26, float %{{.*}})
  %elt.roundeven = call float @llvm.roundeven.f32(float %a)
  ret float %elt.roundeven
}

declare half @llvm.roundeven.f16(half)
declare float @llvm.roundeven.f32(float)
