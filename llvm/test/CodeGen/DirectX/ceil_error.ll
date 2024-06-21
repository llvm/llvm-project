; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation ceil does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload Type

define noundef double @ceil_double(double noundef %a) {
entry:
  %elt.ceil = call double @llvm.ceil.f64(double %a)
  ret double %elt.ceil
}
