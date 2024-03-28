; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation log2 does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload Type

define noundef double @log2_double(double noundef %a) {
entry:
  %elt.log2 = call double @llvm.log2.f64(double %a)
  ret double %elt.log2
}
