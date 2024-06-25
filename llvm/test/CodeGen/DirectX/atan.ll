; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for atan are generated for float and half.

define noundef float @tan_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 17, float %{{.*}})
  %elt.atan = call float @llvm.atan.f32(float %a)
  ret float %elt.atan
}

define noundef half @tan_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 17, half %{{.*}})
  %elt.atan = call half @llvm.atan.f16(half %a)
  ret half %elt.atan
}

declare half @llvm.atan.f16(half)
declare float @llvm.atan.f32(float)
