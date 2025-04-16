; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation dot3 does not support double overload type
; CHECK: in function dot_double3
; CHECK-SAME: Cannot create Dot3 operation: Invalid overload type

define noundef double @dot_double3(double noundef %a1, double noundef %a2,
                                   double noundef %a3, double noundef %b1,
                                   double noundef %b2, double noundef %b3) {
entry:
  %dx.dot = call double @llvm.dx.dot3(double %a1, double %a2, double %a3, double %b1, double %b2, double %b3)
  ret double %dx.dot
}
