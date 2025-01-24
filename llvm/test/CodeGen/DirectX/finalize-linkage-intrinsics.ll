; RUN: opt -S -dxil-finalize-linkage -verify -mtriple=dxil-unknown-shadermodel6.5-library %s

define float @f(float %f) "hlsl.export" {
  %x = call float @llvm.atan.f32(float %f)
  ret float %x
}

declare float @llvm.atan.f32(float)
