; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation cos does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload Type

define noundef double @cos_double(double noundef %a) {
entry:
  %elt.cos = call double @llvm.cos.f64(double %a)
  ret double %elt.cos
}
