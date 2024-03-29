; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for tanh are generated for float and half.

define noundef float @tan_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 20, float %{{.*}})
  %elt.tanh = call float @llvm.tanh.f32(float %a)
  ret float %elt.tanh
}

define noundef half @tan_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 20, half %{{.*}})
  %elt.tanh = call half @llvm.tanh.f16(half %a)
  ret half %elt.tanh
}

declare half @llvm.tanh.f16(half)
declare float @llvm.tanh.f32(float)
