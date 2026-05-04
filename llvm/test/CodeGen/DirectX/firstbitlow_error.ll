; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation firstbitshigh does not support double overload type
; CHECK: intrinsic has incorrect argument type

define noundef double @firstbitlow_double(double noundef %a) {
entry:
  %1 = call double @llvm.dx.firstbitlow.f64(double %a)
  ret double %1
}
