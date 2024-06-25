; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation floor does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload Type

define noundef double @floor_double(double noundef %a) {
entry:
  %elt.floor = call double @llvm.floor.f64(double %a)
  ret double %elt.floor
}
