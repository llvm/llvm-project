; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.0-library %s 2>&1 | FileCheck %s
; The sin intrinsic needs to be scalarized before op lowering

; CHECK: error:
; CHECK-SAME: in function sin_vector
; CHECK-SAME: Cannot create Sin operation: Invalid overload type

define <4 x float> @sin_vector(<4 x float> %a) {
  %x = call <4 x float> @llvm.sin.v4f32(<4 x float> %a)
  ret <4 x float> %x
}
