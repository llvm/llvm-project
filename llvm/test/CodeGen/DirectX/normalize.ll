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

define noundef <2 x half> @test_normalize_half2(<2 x half> noundef %p0) {
entry:
  ; CHECK: extractelement <2 x half> %{{.*}}, i64 0
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: extractelement <2 x half> %{{.*}}, i64 1
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: fadd half %{{.*}}, %{{.*}}
  ; EXPCHECK: call half @llvm.sqrt.f16(half %{{.*}})
  ; DOPCHECK: call half @dx.op.unary.f16(i32 24, half %{{.*}})

  %hlsl.normalize = call <2 x half> @llvm.dx.normalize.v2f16(<2 x half> %p0)
  ret <2 x half> %hlsl.normalize
}

define noundef <3 x half> @test_normalize_half3(<3 x half> noundef %p0) {
entry:
  ; CHECK: extractelement <3 x half> %{{.*}}, i64 0
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: extractelement <3 x half> %{{.*}}, i64 1
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: fadd half %{{.*}}, %{{.*}}
  ; CHECK: extractelement <3 x half> %{{.*}}, i64 2
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: fadd half %{{.*}}, %{{.*}}
  ; EXPCHECK: call half @llvm.sqrt.f16(half %{{.*}})
  ; DOPCHECK: call half @dx.op.unary.f16(i32 24, half %{{.*}})

  %hlsl.normalize = call <3 x half> @llvm.dx.normalize.v3f16(<3 x half> %p0)
  ret <3 x half> %hlsl.normalize
}

define noundef <4 x half> @test_normalize_half4(<4 x half> noundef %p0) {
entry:
  ; CHECK: extractelement <4 x half> %{{.*}}, i64 0
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: extractelement <4 x half> %{{.*}}, i64 1
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: fadd half %{{.*}}, %{{.*}}
  ; CHECK: extractelement <4 x half> %{{.*}}, i64 2
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: fadd half %{{.*}}, %{{.*}}
  ; CHECK: extractelement <4 x half> %{{.*}}, i64 3
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: fadd half %{{.*}}, %{{.*}}
  ; EXPCHECK: call half @llvm.sqrt.f16(half %{{.*}})
  ; DOPCHECK:  call half @dx.op.unary.f16(i32 24, half %{{.*}})

  %hlsl.normalize = call <4 x half> @llvm.dx.normalize.v4f16(<4 x half> %p0)
  ret <4 x half> %hlsl.normalize
}

define noundef <2 x float> @test_normalize_float2(<2 x float> noundef %p0) {
entry:
  ; CHECK: extractelement <2 x float> %{{.*}}, i64 0
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <2 x float> %{{.*}}, i64 1
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; EXPCHECK: call float @llvm.sqrt.f32(float %{{.*}})
  ; DOPCHECK: call float @dx.op.unary.f32(i32 24, float %{{.*}})

  %hlsl.normalize = call <2 x float> @llvm.dx.normalize.v2f32(<2 x float> %p0)
  ret <2 x float> %hlsl.normalize
}

define noundef <3 x float> @test_normalize_float3(<3 x float> noundef %p0) {
entry:
  ; CHECK: extractelement <3 x float> %{{.*}}, i64 0
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <3 x float> %{{.*}}, i64 1
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <3 x float> %{{.*}}, i64 2
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; EXPCHECK: call float @llvm.sqrt.f32(float %{{.*}})
  ; DOPCHECK: call float @dx.op.unary.f32(i32 24, float %{{.*}})

  %hlsl.normalize = call <3 x float> @llvm.dx.normalize.v3f32(<3 x float> %p0)
  ret <3 x float> %hlsl.normalize
}

define noundef <4 x float> @test_normalize_float4(<4 x float> noundef %p0) {
entry:
  ; CHECK: extractelement <4 x float> %{{.*}}, i64 0
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <4 x float> %{{.*}}, i64 1
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <4 x float> %{{.*}}, i64 2
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <4 x float> %{{.*}}, i64 3
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; EXPCHECK: call float @llvm.sqrt.f32(float %{{.*}})
  ; DOPCHECK:  call float @dx.op.unary.f32(i32 24, float %{{.*}})

  %hlsl.normalize = call <4 x float> @llvm.dx.normalize.v4f32(<4 x float> %p0)
  ret <4 x float> %hlsl.normalize
}