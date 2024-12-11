; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation acos does not support double overload type
; CHECK: in function acos_double
; CHECK-SAME: Cannot create ACos operation: Invalid overload type

define noundef double @acos_double(double noundef %a) {
entry:
  %1 = call double @llvm.acos.f64(double %a)
  ret double %1
}
