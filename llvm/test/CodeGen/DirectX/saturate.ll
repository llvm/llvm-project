; RUN: opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; Make sure the intrinsic dx.saturate is to appropriate DXIL op for half/float/double data types.

; CHECK-LABEL: test_saturate_half
define noundef half @test_saturate_half(half noundef %p0) {
entry:
  ; CHECK: call half @dx.op.unary.f16(i32 7, half %p0)
  %hlsl.saturate = call half @llvm.dx.saturate.f16(half %p0)
  ; CHECK: ret half
  ret half %hlsl.saturate
}

; CHECK-LABEL: test_saturate_float
define noundef float @test_saturate_float(float noundef %p0) {
entry:
  ; CHECK: call float @dx.op.unary.f32(i32 7, float %p0)
  %hlsl.saturate = call float @llvm.dx.saturate.f32(float %p0)
  ; CHECK: ret float
  ret float %hlsl.saturate
}

; CHECK-LABEL: test_saturate_double
define noundef double @test_saturate_double(double noundef %p0) {
entry:
  ; CHECK: call double @dx.op.unary.f64(i32 7, double %p0)
  %hlsl.saturate = call double @llvm.dx.saturate.f64(double %p0)
  ; CHECK: ret double
  ret double %hlsl.saturate
}

declare half @llvm.dx.saturate.f16(half)
declare float @llvm.dx.saturate.f32(float)
declare double @llvm.dx.saturate.f64(double)

