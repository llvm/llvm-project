; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation firstbituhigh does not support double overload type
; CHECK: invalid intrinsic signature

define noundef double @firstbituhigh_double(double noundef %a) {
entry:
  %1 = call double @llvm.dx.firstbituhigh.f64(double %a)
  ret double %1
}
