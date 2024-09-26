; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation atan does not support double overload type
; CHECK: in function atan2_double
; CHECK-SAME: Cannot create ATan operation: Invalid overload type

define noundef double @atan2_double(double noundef %a, double noundef %b) #0 {
entry:
  %1 = call double @llvm.atan2.f64(double %a, double %b)
  ret double %1
}
