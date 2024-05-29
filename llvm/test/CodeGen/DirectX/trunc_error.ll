; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation trunc does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload Type

define noundef double @trunc_double(double noundef %a) {
entry:
  %elt.trunc = call double @llvm.trunc.f64(double %a)
  ret double %elt.trunc
}
