; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation dot4 does not support double overload type
; CHECK: in function dot_double4
; CHECK-SAME: Cannot create Dot4 operation: Invalid overload type

define noundef double @dot_double4(<4 x double> noundef %a, <4 x double> noundef %b) {
entry:
  %dx.dot = call double @llvm.dx.dot4.v4f64(<4 x double> %a, <4 x double> %b)
  ret double %dx.dot
}
