; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation dot4 does not support double overload type
; CHECK: in function dot_double4
; CHECK-SAME: Cannot create Dot4 operation: Invalid overload type

define noundef double @dot_double4(double noundef %a1, double noundef %a2,
                                   double noundef %a3, double noundef %a4,
                                   double noundef %b1, double noundef %b2,
                                   double noundef %b3, double noundef %b4) {
entry:
  %dx.dot = call double @llvm.dx.dot4(double %a1, double %a2, double %a3, double %a4, double %b1, double %b2, double %b3, double %b4)
  ret double %dx.dot
}
