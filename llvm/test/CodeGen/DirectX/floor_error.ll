; RUN: not opt -S -dxil-op-lower -mtriple=dxil-pc-shadermodel6.3-library %s 2>&1 | FileCheck %s

; DXIL operation floor does not support double overload type
; CHECK: in function floor_double
; CHECK-SAME: Cannot create Floor operation: Invalid overload type

define noundef double @floor_double(double noundef %a) {
entry:
  %elt.floor = call double @llvm.floor.f64(double %a)
  ret double %elt.floor
}
