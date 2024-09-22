; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S  -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s --check-prefixes=CHECK,DOPCHECK

; Make sure dxil operation function calls for normalize are generated for half/float.

declare half @llvm.dx.normalize.f16(half)
declare <2 x half> @llvm.dx.normalize.v2f16(<2 x half>)
declare <3 x half> @llvm.dx.normalize.v3f16(<3 x half>)
declare <4 x half> @llvm.dx.normalize.v4f16(<4 x half>)

declare float @llvm.dx.normalize.f32(float)
declare <2 x float> @llvm.dx.normalize.v2f32(<2 x float>)
declare <3 x float> @llvm.dx.normalize.v3f32(<3 x float>)
declare <4 x float> @llvm.dx.normalize.v4f32(<4 x float>)

define noundef half @test_normalize_half(half noundef %p0) {
entry:
  ; CHECK: fdiv half %p0, %p0
  %hlsl.normalize = call half @llvm.dx.normalize.f16(half %p0)
  ret half %hlsl.normalize
}

define noundef <2 x half> @test_normalize_half2(<2 x half> noundef %p0) {
entry:
  ; EXPCHECK: [[doth2:%.*]] = call half @llvm.dx.dot2.v2f16(<2 x half> %{{.*}}, <2 x half> %{{.*}})
  ; DOPCHECK: [[doth2:%.*]] = call half @dx.op.dot2.f16(i32 54, half %{{.*}}, half %{{.*}}, half %{{.*}}, half %{{.*}})
  ; EXPCHECK: [[rsqrt:%.*]] = call half @llvm.dx.rsqrt.f16(half [[doth2]])
  ; DOPCHECK: [[rsqrt:%.*]] = call half @dx.op.unary.f16(i32 25, half [[doth2]])
  ; CHECK: [[splatinserth2:%.*]] = insertelement <2 x half> poison, half [[rsqrt]], i64 0
  ; CHECK: [[splat:%.*]] = shufflevector <2 x half> [[splatinserth2]], <2 x half> poison, <2 x i32> zeroinitializer
  ; CHECK: fmul <2 x half> %p0, [[splat]]

  %hlsl.normalize = call <2 x half> @llvm.dx.normalize.v2f16(<2 x half> %p0)
  ret <2 x half> %hlsl.normalize
}

define noundef <3 x half> @test_normalize_half3(<3 x half> noundef %p0) {
entry:
  ; EXPCHECK: [[doth3:%.*]] = call half @llvm.dx.dot3.v3f16(<3 x half> %{{.*}}, <3 x half> %{{.*}})
  ; DOPCHECK: [[doth3:%.*]] = call half @dx.op.dot3.f16(i32 55, half %{{.*}}, half %{{.*}}, half %{{.*}}, half %{{.*}})
  ; EXPCHECK: [[rsqrt:%.*]] = call half @llvm.dx.rsqrt.f16(half [[doth3]])
  ; DOPCHECK: [[rsqrt:%.*]] = call half @dx.op.unary.f16(i32 25, half [[doth3]])
  ; CHECK: [[splatinserth3:%.*]] = insertelement <3 x half> poison, half [[rsqrt]], i64 0
  ; CHECK: [[splat:%.*]] shufflevector <3 x half> [[splatinserth3]], <3 x half> poison, <3 x i32> zeroinitializer
  ; CHECK: fmul <3 x half> %p0, %.splat

  %hlsl.normalize = call <3 x half> @llvm.dx.normalize.v3f16(<3 x half> %p0)
  ret <3 x half> %hlsl.normalize
}

define noundef <4 x half> @test_normalize_half4(<4 x half> noundef %p0) {
entry:
  ; EXPCHECK: [[doth4:%.*]] = call half @llvm.dx.dot4.v4f16(<4 x half> %{{.*}}, <4 x half> %{{.*}})
  ; DOPCHECK: [[doth4:%.*]] = call half @dx.op.dot4.f16(i32 56, half %{{.*}}, half %{{.*}}, half %{{.*}}, half %{{.*}})
  ; EXPCHECK: [[rsqrt:%.*]] = call half @llvm.dx.rsqrt.f16(half [[doth4]])
  ; DOPCHECK: [[rsqrt:%.*]] = call half @dx.op.unary.f16(i32 25, half [[doth4]])
  ; CHECK: [[splatinserth4:%.*]] = insertelement <4 x half> poison, half [[rsqrt]], i64 0
  ; CHECK: [[splat:%.*]] shufflevector <4 x half> [[splatinserth4]], <4 x half> poison, <4 x i32> zeroinitializer
  ; CHECK: fmul <4 x half> %p0, %.splat

  %hlsl.normalize = call <4 x half> @llvm.dx.normalize.v4f16(<4 x half> %p0)
  ret <4 x half> %hlsl.normalize
}

define noundef float @test_normalize_float(float noundef %p0) {
entry:
  ; CHECK: fdiv float %p0, %p0
  %hlsl.normalize = call float @llvm.dx.normalize.f32(float %p0)
  ret float %hlsl.normalize
}

define noundef <2 x float> @test_normalize_float2(<2 x float> noundef %p0) {
entry:
  ; EXPCHECK: [[dotf2:%.*]] = call float @llvm.dx.dot2.v2f32(<2 x float> %{{.*}}, <2 x float> %{{.*}})
  ; DOPCHECK: [[dotf2:%.*]] = call float @dx.op.dot2.f32(i32 54, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
  ; EXPCHECK: [[rsqrt:%.*]] = call float @llvm.dx.rsqrt.f32(float [[dotf2]])
  ; DOPCHECK: [[rsqrt:%.*]] = call float @dx.op.unary.f32(i32 25, float [[dotf2]])
  ; CHECK: [[splatinsertf2:%.*]] = insertelement <2 x float> poison, float [[rsqrt]], i64 0
  ; CHECK: [[splat:%.*]] shufflevector <2 x float> [[splatinsertf2]], <2 x float> poison, <2 x i32> zeroinitializer
  ; CHECK: fmul <2 x float> %p0, %.splat

  %hlsl.normalize = call <2 x float> @llvm.dx.normalize.v2f32(<2 x float> %p0)
  ret <2 x float> %hlsl.normalize
}

define noundef <3 x float> @test_normalize_float3(<3 x float> noundef %p0) {
entry:
  ; EXPCHECK: [[dotf3:%.*]] = call float @llvm.dx.dot3.v3f32(<3 x float> %{{.*}}, <3 x float> %{{.*}})
  ; DOPCHECK: [[dotf3:%.*]] = call float @dx.op.dot3.f32(i32 55, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
  ; EXPCHECK: [[rsqrt:%.*]] = call float @llvm.dx.rsqrt.f32(float [[dotf3]])
  ; DOPCHECK: [[rsqrt:%.*]] = call float @dx.op.unary.f32(i32 25, float [[dotf3]])
  ; CHECK: [[splatinsertf3:%.*]] = insertelement <3 x float> poison, float [[rsqrt]], i64 0
  ; CHECK: [[splat:%.*]] shufflevector <3 x float> [[splatinsertf3]], <3 x float> poison, <3 x i32> zeroinitializer
  ; CHECK: fmul <3 x float> %p0, %.splat

  %hlsl.normalize = call <3 x float> @llvm.dx.normalize.v3f32(<3 x float> %p0)
  ret <3 x float> %hlsl.normalize
}

define noundef <4 x float> @test_normalize_float4(<4 x float> noundef %p0) {
entry:
  ; EXPCHECK: [[dotf4:%.*]] = call float @llvm.dx.dot4.v4f32(<4 x float> %{{.*}}, <4 x float> %{{.*}})
  ; DOPCHECK: [[dotf4:%.*]] = call float @dx.op.dot4.f32(i32 56, float %{{.*}}, float %{{.*}}, float %{{.*}}, float %{{.*}})
  ; EXPCHECK: [[rsqrt:%.*]] = call float @llvm.dx.rsqrt.f32(float [[dotf4]])
  ; DOPCHECK: [[rsqrt:%.*]] = call float @dx.op.unary.f32(i32 25, float [[dotf4]])
  ; CHECK: [[splatinsertf4:%.*]] = insertelement <4 x float> poison, float [[rsqrt]], i64 0
  ; CHECK: [[splat:%.*]] shufflevector <4 x float> [[splatinsertf4]], <4 x float> poison, <4 x i32> zeroinitializer
  ; CHECK: fmul <4 x float> %p0, %.splat

  %hlsl.normalize = call <4 x float> @llvm.dx.normalize.v4f32(<4 x float> %p0)
  ret <4 x float> %hlsl.normalize
}
