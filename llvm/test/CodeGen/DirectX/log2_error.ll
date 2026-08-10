; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation log2 does not support double overload type
; CHECK: in function log2_double
; CHECK-SAME: Cannot create Log2 operation: Invalid overload type

define noundef double @log2_double(double noundef %a) {
entry:
  %elt.log2 = call double @llvm.log2.f64(double %a)
  ret double %elt.log2
}
