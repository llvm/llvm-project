; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s --check-prefix=CHECK
; RUN: opt -S  -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library < %s | FileCheck %s --check-prefix=CHECK

; Make sure dxil operation function calls for cross are generated for half/float.

declare <3 x half> @llvm.dx.cross.v3f16(<3 x half>, <3 x half>)
declare <3 x float> @llvm.dx.cross.v3f32(<3 x float>, <3 x float>)

define noundef <3 x half> @test_cross_half3(<3 x half> noundef %p0, <3 x half> noundef %p1) {
entry:
  ; CHECK: %0 = extractelement <3 x half> %p0, i64 0
  ; CHECK: %1 = extractelement <3 x half> %p0, i64 1
  ; CHECK: %2 = extractelement <3 x half> %p0, i64 2
  ; CHECK: %3 = extractelement <3 x half> %p1, i64 0
  ; CHECK: %4 = extractelement <3 x half> %p1, i64 1
  ; CHECK: %5 = extractelement <3 x half> %p1, i64 2
  ; CHECK: %6 = fmul half %1, %5
  ; CHECK: %7 = fmul half %2, %4
  ; CHECK: %8 = fsub half %6, %7
  ; CHECK: %9 = fmul half %2, %3
  ; CHECK: %10 = fmul half %0, %5
  ; CHECK: %11 = fsub half %9, %10
  ; CHECK: %12 = fmul half %0, %4
  ; CHECK: %13 = fmul half %1, %3
  ; CHECK: %14 = fsub half %12, %13
  ; CHECK: %15 = insertelement <3 x half> undef, half %8, i64 0
  ; CHECK: %16 = insertelement <3 x half> %15, half %11, i64 1
  ; CHECK: %17 = insertelement <3 x half> %16, half %14, i64 2
  ; CHECK: ret <3 x half> %17
  %hlsl.cross = call <3 x half> @llvm.dx.cross.v3f16(<3 x half> %p0, <3 x half> %p1)
  ret <3 x half> %hlsl.cross
}

define noundef <3 x float> @test_cross_float3(<3 x float> noundef %p0, <3 x float> noundef %p1) {
entry:
  ; CHECK: %0 = extractelement <3 x float> %p0, i64 0
  ; CHECK: %1 = extractelement <3 x float> %p0, i64 1
  ; CHECK: %2 = extractelement <3 x float> %p0, i64 2
  ; CHECK: %3 = extractelement <3 x float> %p1, i64 0
  ; CHECK: %4 = extractelement <3 x float> %p1, i64 1
  ; CHECK: %5 = extractelement <3 x float> %p1, i64 2
  ; CHECK: %6 = fmul float %1, %5
  ; CHECK: %7 = fmul float %2, %4
  ; CHECK: %8 = fsub float %6, %7
  ; CHECK: %9 = fmul float %2, %3
  ; CHECK: %10 = fmul float %0, %5
  ; CHECK: %11 = fsub float %9, %10
  ; CHECK: %12 = fmul float %0, %4
  ; CHECK: %13 = fmul float %1, %3
  ; CHECK: %14 = fsub float %12, %13
  ; CHECK: %15 = insertelement <3 x float> undef, float %8, i64 0
  ; CHECK: %16 = insertelement <3 x float> %15, float %11, i64 1
  ; CHECK: %17 = insertelement <3 x float> %16, float %14, i64 2
  ; CHECK: ret <3 x float> %17
  %hlsl.cross = call <3 x float> @llvm.dx.cross.v3f32(<3 x float> %p0, <3 x float> %p1)
  ret <3 x float> %hlsl.cross
}
