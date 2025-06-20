; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation ceil does not support double overload type
; CHECK: in function ceil_double
; CHECK-SAME: Cannot create Ceil operation: Invalid overload type

define noundef double @ceil_double(double noundef %a) {
entry:
  %elt.ceil = call double @llvm.ceil.f64(double %a)
  ret double %elt.ceil
}
