; RUN: not opt -S -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation length does not support 1-element vector types.
; CHECK: LLVM ERROR: Invalid input type for length intrinsic

define noundef float @test_length_float(<1 x float> noundef %p0) {
entry:
  %hlsl.length = call float @llvm.dx.length.v1f32(<1 x float> %p0)
  ret float %hlsl.length
}
