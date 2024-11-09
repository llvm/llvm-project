; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s --check-prefixes=CHECK,EXPCHECK
; RUN: opt -S  -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s --check-prefixes=CHECK,DOPCHECK

; Make sure dxil operation function calls for length are generated for half/float.

declare half @llvm.fabs.f16(half)
declare half @llvm.dx.length.v2f16(<2 x half>)
declare half @llvm.dx.length.v3f16(<3 x half>)
declare half @llvm.dx.length.v4f16(<4 x half>)

declare float @llvm.fabs.f32(float)
declare float @llvm.dx.length.v2f32(<2 x float>)
declare float @llvm.dx.length.v3f32(<3 x float>)
declare float @llvm.dx.length.v4f32(<4 x float>)

define noundef half @test_length_half2(<2 x half> noundef %p0) {
entry:
  ; CHECK: extractelement <2 x half> %{{.*}}, i64 0
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: extractelement <2 x half> %{{.*}}, i64 1
  ; CHECK: fmul half %{{.*}}, %{{.*}}
  ; CHECK: fadd half %{{.*}}, %{{.*}}
  ; EXPCHECK: call half @llvm.sqrt.f16(half %{{.*}})
  ; DOPCHECK: call half @dx.op.unary.f16(i32 24, half %{{.*}})

  %hlsl.length = call half @llvm.dx.length.v2f16(<2 x half> %p0)
  ret half %hlsl.length
}

define noundef half @test_length_half3(<3 x half> noundef %p0) {
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

  %hlsl.length = call half @llvm.dx.length.v3f16(<3 x half> %p0)
  ret half %hlsl.length
}

define noundef half @test_length_half4(<4 x half> noundef %p0) {
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

  %hlsl.length = call half @llvm.dx.length.v4f16(<4 x half> %p0)
  ret half %hlsl.length
}

define noundef float @test_length_float2(<2 x float> noundef %p0) {
entry:
  ; CHECK: extractelement <2 x float> %{{.*}}, i64 0
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: extractelement <2 x float> %{{.*}}, i64 1
  ; CHECK: fmul float %{{.*}}, %{{.*}}
  ; CHECK: fadd float %{{.*}}, %{{.*}}
  ; EXPCHECK: call float @llvm.sqrt.f32(float %{{.*}})
  ; DOPCHECK: call float @dx.op.unary.f32(i32 24, float %{{.*}})

  %hlsl.length = call float @llvm.dx.length.v2f32(<2 x float> %p0)
  ret float %hlsl.length
}

define noundef float @test_length_float3(<3 x float> noundef %p0) {
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

  %hlsl.length = call float @llvm.dx.length.v3f32(<3 x float> %p0)
  ret float %hlsl.length
}

define noundef float @test_length_float4(<4 x float> noundef %p0) {
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

  %hlsl.length = call float @llvm.dx.length.v4f32(<4 x float> %p0)
  ret float %hlsl.length
}
