; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation atan does not support double overload type
; CHECK: in function atan_double
; CHECK-SAME: Cannot create ATan operation: Invalid overload type

define noundef double @atan_double(double noundef %a) {
entry:
  %1 = call double @llvm.atan.f64(double %a)
  ret double %1
}
