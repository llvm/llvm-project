; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for ceil are generated for float and half.

define noundef float @ceil_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 28, float %{{.*}})
  %elt.ceil = call float @llvm.ceil.f32(float %a)
  ret float %elt.ceil
}

define noundef half @ceil_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 28, half %{{.*}})
  %elt.ceil = call half @llvm.ceil.f16(half %a)
  ret half %elt.ceil
}

declare half @llvm.ceil.f16(half)
declare float @llvm.ceil.f32(float)
