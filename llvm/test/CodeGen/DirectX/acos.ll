; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for acos are generated for float and half.

define noundef float @tan_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 15, float %{{.*}})
  %elt.acos = call float @llvm.acos.f32(float %a)
  ret float %elt.acos
}

define noundef half @tan_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 15, half %{{.*}})
  %elt.acos = call half @llvm.acos.f16(half %a)
  ret half %elt.acos
}

declare half @llvm.acos.f16(half)
declare float @llvm.acos.f32(float)
