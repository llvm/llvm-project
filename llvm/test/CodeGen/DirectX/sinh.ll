; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for sinh are generated for float and half.

define noundef float @tan_float(float noundef %a) {
entry:
; CHECK:call float @dx.op.unary.f32(i32 19, float %{{.*}})
  %elt.sinh = call float @llvm.sinh.f32(float %a)
  ret float %elt.sinh
}

define noundef half @tan_half(half noundef %a) {
entry:
; CHECK:call half @dx.op.unary.f16(i32 19, half %{{.*}})
  %elt.sinh = call half @llvm.sinh.f16(half %a)
  ret half %elt.sinh
}

declare half @llvm.sinh.f16(half)
declare float @llvm.sinh.f32(float)
