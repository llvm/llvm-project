; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt -S  -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s --check-prefix=CHECK

; Make sure dxil operation function calls for step are generated for half/float.

declare half @llvm.dx.step.f16(half, half)
declare <2 x half> @llvm.dx.step.v2f16(<2 x half>, <2 x half>)
declare <3 x half> @llvm.dx.step.v3f16(<3 x half>, <3 x half>)
declare <4 x half> @llvm.dx.step.v4f16(<4 x half>, <4 x half>)

declare float @llvm.dx.step.f32(float, float)
declare <2 x float> @llvm.dx.step.v2f32(<2 x float>, <2 x float>)
declare <3 x float> @llvm.dx.step.v3f32(<3 x float>, <3 x float>)
declare <4 x float> @llvm.dx.step.v4f32(<4 x float>, <4 x float>)

define noundef half @test_step_half(half noundef %p0, half noundef %p1) {
entry:
  ; CHECK: %0 = fcmp olt half %p1, %p0
  ; CHECK: %1 = select i1 %0, half 0xH0000, half 0xH3C00
  %hlsl.step = call half @llvm.dx.step.f16(half %p0, half %p1)
  ret half %hlsl.step
}

define noundef <2 x half> @test_step_half2(<2 x half> noundef %p0, <2 x half> noundef %p1) {
entry:
  ; CHECK: %0 = fcmp olt <2 x half> %p1, %p0
  ; CHECK: %1 = select <2 x i1> %0, <2 x half> zeroinitializer, <2 x half> splat (half 0xH3C00)
  %hlsl.step = call <2 x half> @llvm.dx.step.v2f16(<2 x half> %p0, <2 x half> %p1)
  ret <2 x half> %hlsl.step
}

define noundef <3 x half> @test_step_half3(<3 x half> noundef %p0, <3 x half> noundef %p1) {
entry:
  ; CHECK: %0 = fcmp olt <3 x half> %p1, %p0
  ; CHECK: %1 = select <3 x i1> %0, <3 x half> zeroinitializer, <3 x half> splat (half 0xH3C00)
  %hlsl.step = call <3 x half> @llvm.dx.step.v3f16(<3 x half> %p0, <3 x half> %p1)
  ret <3 x half> %hlsl.step
}

define noundef <4 x half> @test_step_half4(<4 x half> noundef %p0, <4 x half> noundef %p1) {
entry:
  ; CHECK: %0 = fcmp olt <4 x half> %p1, %p0
  ; CHECK: %1 = select <4 x i1> %0, <4 x half> zeroinitializer, <4 x half> splat (half 0xH3C00)
  %hlsl.step = call <4 x half> @llvm.dx.step.v4f16(<4 x half> %p0, <4 x half> %p1)
  ret <4 x half> %hlsl.step
}

define noundef float @test_step_float(float noundef %p0, float noundef %p1) {
entry:
  ; CHECK: %0 = fcmp olt float %p1, %p0
  ; CHECK: %1 = select i1 %0, float 0.000000e+00, float 1.000000e+00
  %hlsl.step = call float @llvm.dx.step.f32(float %p0, float %p1)
  ret float %hlsl.step
}

define noundef <2 x float> @test_step_float2(<2 x float> noundef %p0, <2 x float> noundef %p1) {
entry:
  ; CHECK: %0 = fcmp olt <2 x float> %p1, %p0
  ; CHECK: %1 = select <2 x i1> %0, <2 x float> zeroinitializer, <2 x float> splat (float 1.000000e+00)
  %hlsl.step = call <2 x float> @llvm.dx.step.v2f32(<2 x float> %p0, <2 x float> %p1)
  ret <2 x float> %hlsl.step
}

define noundef <3 x float> @test_step_float3(<3 x float> noundef %p0, <3 x float> noundef %p1) {
entry:
  ; CHECK: %0 = fcmp olt <3 x float> %p1, %p0
  ; CHECK: %1 = select <3 x i1> %0, <3 x float> zeroinitializer, <3 x float> splat (float 1.000000e+00)
  %hlsl.step = call <3 x float> @llvm.dx.step.v3f32(<3 x float> %p0, <3 x float> %p1)
  ret <3 x float> %hlsl.step
}

define noundef <4 x float> @test_step_float4(<4 x float> noundef %p0, <4 x float> noundef %p1) {
entry:
  ; CHECK: %0 = fcmp olt <4 x float> %p1, %p0
  ; CHECK: %1 = select <4 x i1> %0, <4 x float> zeroinitializer, <4 x float> splat (float 1.000000e+00)
  %hlsl.step = call <4 x float> @llvm.dx.step.v4f32(<4 x float> %p0, <4 x float> %p1)
  ret <4 x float> %hlsl.step
}
