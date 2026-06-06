; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation cosh does not support double overload type
; CHECK: in function cosh_double
; CHECK-SAME: Cannot create HCos operation: Invalid overload type

define noundef double @cosh_double(double noundef %a) {
entry:
  %1 = call double @llvm.cosh.f64(double %a)
  ret double %1
}
