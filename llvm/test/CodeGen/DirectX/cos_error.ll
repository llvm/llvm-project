; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation cos does not support double overload type
; CHECK: in function cos_double
; CHECK-SAME: Cannot create Cos operation: Invalid overload type

define noundef double @cos_double(double noundef %a) {
entry:
  %elt.cos = call double @llvm.cos.f64(double %a)
  ret double %elt.cos
}
