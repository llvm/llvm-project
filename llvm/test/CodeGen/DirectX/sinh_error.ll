; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation sinh does not support double overload type
; CHECK: in function sinh_double
; CHECK-SAME: Cannot create HSin operation: Invalid overload type

define noundef double @sinh_double(double noundef %a) {
entry:
  %1 = call double @llvm.sinh.f64(double %a)
  ret double %1
}
