; RUN: opt -S -dxil-op-lower < %s | FileCheck %s

; Make sure dxil operation function calls for cos are generated for float and half.

define noundef half @test_cos_half(half noundef %a) #0 {
entry:
  ; CHECK: call half @dx.op.unary.f16(i32 12, half %{{.*}})
  %1 = call half @llvm.cos.f16(half %a)
  ret half %1
}

define noundef float @test_cos_float(float noundef %a) #0 {
entry:
  ; CHECK: call float @dx.op.unary.f32(i32 12, float %{{.*}})
  %1 = call float @llvm.cos.f32(float %a)
  ret float %1
}
