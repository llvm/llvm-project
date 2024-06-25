; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for sqrt are generated for float and half.

define noundef float @sqrt_float(float noundef %a) #0 {
entry:
; CHECK:call float @dx.op.unary.f32(i32 24, float %{{.*}})
  %elt.sqrt = call float @llvm.sqrt.f32(float %a)
  ret float %elt.sqrt
}

define noundef half @sqrt_half(half noundef %a) #0 {
entry:
; CHECK:call half @dx.op.unary.f16(i32 24, half %{{.*}})
  %elt.sqrt = call half @llvm.sqrt.f16(half %a)
  ret half %elt.sqrt
}

declare half @llvm.sqrt.f16(half)
declare float @llvm.sqrt.f32(float)
