; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for floor are generated for half and float.

define noundef half @test_floor_half(half noundef %a) #0 {
entry:
  ; CHECK: call half @dx.op.unary.f16(i32 27, half %{{.*}})
  %1 = call half @llvm.floor.f16(half %a)
  ret half %1
}

define noundef float @test_floor_float(float noundef %a) #0 {
entry:
  ; CHECK: call float @dx.op.unary.f32(i32 27, float %{{.*}})
  %1 = call float @llvm.floor.f32(float %a)
  ret float %1
}
