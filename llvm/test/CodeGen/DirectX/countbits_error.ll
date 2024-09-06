; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation ctpop does not support double overload type
; CHECK: invalid intrinsic signature

define noundef double @countbits_double(double noundef %a) {
entry:
  %elt.ctpop = call double @llvm.ctpop.f64(double %a)
  ret double %elt.ctpop
}
