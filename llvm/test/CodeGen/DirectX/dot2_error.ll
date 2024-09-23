; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation dot2 does not support double overload type
; CHECK: in function dot_double2
; CHECK-SAME: Cannot create Dot2 operation: Invalid overload type

define noundef double @dot_double2(<2 x double> noundef %a, <2 x double> noundef %b) {
entry:
  %dx.dot = call double @llvm.dx.dot2.v2f64(<2 x double> %a, <2 x double> %b)
  ret double %dx.dot
}
