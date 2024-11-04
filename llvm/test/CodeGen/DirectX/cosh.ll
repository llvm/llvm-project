; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for cosh are generated for float and half.

define noundef float @tan_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 18, float %{{.*}})
  %elt.cosh = call float @llvm.cosh.f32(float %a)
  ret float %elt.cosh
}

define noundef half @tan_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 18, half %{{.*}})
  %elt.cosh = call half @llvm.cosh.f16(half %a)
  ret half %elt.cosh
}

declare half @llvm.cosh.f16(half)
declare float @llvm.cosh.f32(float)
