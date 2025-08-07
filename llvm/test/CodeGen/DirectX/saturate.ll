; RUN: opt -S -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s
; Make sure the intrinsic dx.saturate is to appropriate DXIL op for half/float/double data types.

; CHECK-LABEL: test_saturate_half
define noundef half @test_saturate_half(half noundef %p0) {
entry:
  ; CHECK: call half @dx.op.unary.f16(i32 7, half %p0) #[[#ATTR:]]
  %hlsl.saturate = call half @llvm.dx.saturate.f16(half %p0)
  ; CHECK: ret half
  ret half %hlsl.saturate
}

; CHECK-LABEL: test_saturate_float
define noundef float @test_saturate_float(float noundef %p0) {
entry:
  ; CHECK: call float @dx.op.unary.f32(i32 7, float %p0) #[[#ATTR]]
  %hlsl.saturate = call float @llvm.dx.saturate.f32(float %p0)
  ; CHECK: ret float
  ret float %hlsl.saturate
}

; CHECK-LABEL: test_saturate_double
define noundef double @test_saturate_double(double noundef %p0) {
entry:
  ; CHECK: call double @dx.op.unary.f64(i32 7, double %p0) #[[#ATTR]]
  %hlsl.saturate = call double @llvm.dx.saturate.f64(double %p0)
  ; CHECK: ret double
  ret double %hlsl.saturate
}

; CHECK-LABEL: test_saturate_half4
define noundef <4 x half> @test_saturate_half4(<4 x half> noundef %p0) {
entry:
  ; CHECK: call half @dx.op.unary.f16(i32 7, half
  ; CHECK: call half @dx.op.unary.f16(i32 7, half
  ; CHECK: call half @dx.op.unary.f16(i32 7, half
  ; CHECK: call half @dx.op.unary.f16(i32 7, half
  %hlsl.saturate = call <4 x half> @llvm.dx.saturate.v4f16(<4 x half> %p0)
  ret <4 x half> %hlsl.saturate
}

; CHECK-LABEL: test_saturate_float3
define noundef <3 x float> @test_saturate_float3(<3 x float> noundef %p0) {
entry:
  ; CHECK: call float @dx.op.unary.f32(i32 7, float
  ; CHECK: call float @dx.op.unary.f32(i32 7, float
  ; CHECK: call float @dx.op.unary.f32(i32 7, float
  %hlsl.saturate = call <3 x float> @llvm.dx.saturate.v3f32(<3 x float> %p0)
  ret <3 x float> %hlsl.saturate
}

; CHECK-LABEL: test_saturate_double2
define noundef <2 x double> @test_saturate_double2(<2 x double> noundef %p0) {
entry:
  ; CHECK: call double @dx.op.unary.f64(i32 7, double
  ; CHECK: call double @dx.op.unary.f64(i32 7, double
  %hlsl.saturate = call <2 x double> @llvm.dx.saturate.v4f64(<2 x double> %p0)
  ret <2 x double> %hlsl.saturate
}


; CHECK: attributes #[[#ATTR]] = {{{.*}} memory(none) {{.*}}}
