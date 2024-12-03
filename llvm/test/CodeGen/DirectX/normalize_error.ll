; RUN: not opt -S -dxil-intrinsic-expansion -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation normalize does not support double overload type
; CHECK: Cannot create Dot2 operation: Invalid overload type

define noundef <2 x double> @test_normalize_double2(<2 x double> noundef %p0) {
entry:
  %hlsl.normalize = call <2 x double> @llvm.dx.normalize.v2f32(<2 x double> %p0)
  ret <2 x double> %hlsl.normalize
}
