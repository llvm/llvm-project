; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation sqrt does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload Type

define noundef double @sqrt_double(double noundef %a) {
entry:
  %elt.sqrt = call double @llvm.sqrt.f64(double %a)
  ret double %elt.sqrt
}
