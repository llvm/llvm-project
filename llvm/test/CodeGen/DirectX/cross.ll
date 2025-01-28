; RUN: opt -S  -dxil-intrinsic-expansion  < %s | FileCheck %s

; Make sure dxil operation function calls for cross are generated for half/float.

declare <3 x half> @llvm.dx.cross.v3f16(<3 x half>, <3 x half>)
declare <3 x float> @llvm.dx.cross.v3f32(<3 x float>, <3 x float>)

define noundef <3 x half> @test_cross_half3(<3 x half> noundef %p0, <3 x half> noundef %p1) {
entry:
  ; CHECK: %x0 = extractelement <3 x half> %p0, i64 0
  ; CHECK: %x1 = extractelement <3 x half> %p0, i64 1
  ; CHECK: %x2 = extractelement <3 x half> %p0, i64 2
  ; CHECK: %y0 = extractelement <3 x half> %p1, i64 0
  ; CHECK: %y1 = extractelement <3 x half> %p1, i64 1
  ; CHECK: %y2 = extractelement <3 x half> %p1, i64 2
  ; CHECK: %0 = fmul half %x1, %y2
  ; CHECK: %1 = fmul half %x2, %y1
  ; CHECK: %hlsl.cross1 = fsub half %0, %1
  ; CHECK: %2 = fmul half %x2, %y0
  ; CHECK: %3 = fmul half %x0, %y2
  ; CHECK: %hlsl.cross2 = fsub half %2, %3
  ; CHECK: %4 = fmul half %x0, %y1
  ; CHECK: %5 = fmul half %x1, %y0
  ; CHECK: %hlsl.cross3 = fsub half %4, %5
  ; CHECK: %6 = insertelement <3 x half> undef, half %hlsl.cross1, i64 0
  ; CHECK: %7 = insertelement <3 x half> %6, half %hlsl.cross2, i64 1
  ; CHECK: %8 = insertelement <3 x half> %7, half %hlsl.cross3, i64 2
  ; CHECK: ret <3 x half> %8
  %hlsl.cross = call <3 x half> @llvm.dx.cross.v3f16(<3 x half> %p0, <3 x half> %p1)
  ret <3 x half> %hlsl.cross
}

define noundef <3 x float> @test_cross_float3(<3 x float> noundef %p0, <3 x float> noundef %p1) {
entry:
  ; CHECK: %x0 = extractelement <3 x float> %p0, i64 0
  ; CHECK: %x1 = extractelement <3 x float> %p0, i64 1
  ; CHECK: %x2 = extractelement <3 x float> %p0, i64 2
  ; CHECK: %y0 = extractelement <3 x float> %p1, i64 0
  ; CHECK: %y1 = extractelement <3 x float> %p1, i64 1
  ; CHECK: %y2 = extractelement <3 x float> %p1, i64 2
  ; CHECK: %0 = fmul float %x1, %y2
  ; CHECK: %1 = fmul float %x2, %y1
  ; CHECK: %hlsl.cross1 = fsub float %0, %1
  ; CHECK: %2 = fmul float %x2, %y0
  ; CHECK: %3 = fmul float %x0, %y2
  ; CHECK: %hlsl.cross2 = fsub float %2, %3
  ; CHECK: %4 = fmul float %x0, %y1
  ; CHECK: %5 = fmul float %x1, %y0
  ; CHECK: %hlsl.cross3 = fsub float %4, %5
  ; CHECK: %6 = insertelement <3 x float> undef, float %hlsl.cross1, i64 0
  ; CHECK: %7 = insertelement <3 x float> %6, float %hlsl.cross2, i64 1
  ; CHECK: %8 = insertelement <3 x float> %7, float %hlsl.cross3, i64 2
  ; CHECK: ret <3 x float> %8
  %hlsl.cross = call <3 x float> @llvm.dx.cross.v3f32(<3 x float> %p0, <3 x float> %p1)
  ret <3 x float> %hlsl.cross
}
