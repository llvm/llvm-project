; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for cos are generated for float and half.

define noundef float @cos_float(float noundef %a) #0 {
entry:
; CHECK:call float @dx.op.unary.f32(i32 12, float %{{.*}})
  %elt.cos = call float @llvm.cos.f32(float %a)
  ret float %elt.cos
}

define noundef half @cos_half(half noundef %a) #0 {
entry:
; CHECK:call half @dx.op.unary.f16(i32 12, half %{{.*}})
  %elt.cos = call half @llvm.cos.f16(half %a)
  ret half %elt.cos
}

declare half @llvm.cos.f16(half)
declare float @llvm.cos.f32(float)
