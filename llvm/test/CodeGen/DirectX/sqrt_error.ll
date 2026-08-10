; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation sqrt does not support double overload type
; CHECK: in function sqrt_double
; CHECK-SAME: Cannot create Sqrt operation: Invalid overload type

define noundef double @sqrt_double(double noundef %a) {
entry:
  %elt.sqrt = call double @llvm.sqrt.f64(double %a)
  ret double %elt.sqrt
}
