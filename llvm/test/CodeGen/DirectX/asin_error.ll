; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation asin does not support double overload type
; CHECK: in function asin_double
; CHECK-SAME: Cannot create ASin operation: Invalid overload type

define noundef double @asin_double(double noundef %a) {
entry:
  %1 = call double @llvm.asin.f64(double %a)
  ret double %1
}
