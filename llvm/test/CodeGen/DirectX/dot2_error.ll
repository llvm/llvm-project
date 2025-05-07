; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation dot2 does not support double overload type
; CHECK: in function dot_double2
; CHECK-SAME: Cannot create Dot2 operation: Invalid overload type

define noundef double @dot_double2(double noundef %a1, double noundef %a2,
                                   double noundef %b1, double noundef %b2) {
entry:
  %dx.dot = call double @llvm.dx.dot2(double %a1, double %a2, double %b1, double %b2)
  ret double %dx.dot
}
