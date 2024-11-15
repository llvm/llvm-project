; RUN: not opt -S -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation length does not support double overload type
; CHECK: Cannot create Sqrt operation: Invalid overload type

define noundef double @test_length_double2(<2 x double> noundef %p0) {
entry:
  %hlsl.length = call double @llvm.dx.length.v2f32(<2 x double> %p0)
  ret double %hlsl.length
}
