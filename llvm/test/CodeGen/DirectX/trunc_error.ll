; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation trunc does not support double overload type
; CHECK: in function trunc_double
; CHECK-SAME: Cannot create Trunc operation: Invalid overload type

define noundef double @trunc_double(double noundef %a) {
entry:
  %elt.trunc = call double @llvm.trunc.f64(double %a)
  ret double %elt.trunc
}
