; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for tan are generated for float and half.

define noundef float @tan_float(float noundef %a) #0 {
entry:
; CHECK:call float @dx.op.unary.f32(i32 14, float %{{.*}})
  %elt.tan = call float @llvm.tan.f32(float %a)
  ret float %elt.tan
}

define noundef half @tan_half(half noundef %a) #0 {
entry:
; CHECK:call half @dx.op.unary.f16(i32 14, half %{{.*}})
  %elt.tan = call half @llvm.tan.f16(half %a)
  ret half %elt.tan
}

declare half @llvm.tan.f16(half)
declare float @llvm.tan.f32(float)
