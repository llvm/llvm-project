; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for trunc are generated for float and half.

define noundef float @trunc_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 29, float %{{.*}})
  %elt.trunc = call float @llvm.trunc.f32(float %a)
  ret float %elt.trunc
}

define noundef half @trunc_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 29, half %{{.*}})
  %elt.trunc = call half @llvm.trunc.f16(half %a)
  ret half %elt.trunc
}

declare half @llvm.trunc.f16(half)
declare float @llvm.trunc.f32(float)
