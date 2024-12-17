; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation length does not support scalar types
; CHECK: error: invalid intrinsic signature

define noundef float @test_length_float(float noundef %p0) {
entry:
  %hlsl.length = call float @llvm.dx.length.f32(float %p0)
  ret float %hlsl.length
}
