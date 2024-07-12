; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for asin are generated for float and half.

define noundef float @tan_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 16, float %{{.*}})
  %elt.asin = call float @llvm.asin.f32(float %a)
  ret float %elt.asin
}

define noundef half @tan_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 16, half %{{.*}})
  %elt.asin = call half @llvm.asin.f16(half %a)
  ret half %elt.asin
}

declare half @llvm.asin.f16(half)
declare float @llvm.asin.f32(float)
