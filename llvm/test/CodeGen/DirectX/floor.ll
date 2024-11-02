; RUN: opt -S -passes=dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; Make sure dxil operation function calls for floor are generated for float and half.

define noundef float @floor_float(float noundef %a) #0 {
entry:
; CHECK:call float @dx.op.unary.f32(i32 27, float %{{.*}})
  %elt.floor = call float @llvm.floor.f32(float %a)
  ret float %elt.floor
}

define noundef half @floor_half(half noundef %a) #0 {
entry:
; CHECK:call half @dx.op.unary.f16(i32 27, half %{{.*}})
  %elt.floor = call half @llvm.floor.f16(half %a)
  ret half %elt.floor
}

declare half @llvm.floor.f16(half)
declare float @llvm.floor.f32(float)
