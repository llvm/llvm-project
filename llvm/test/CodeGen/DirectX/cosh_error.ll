; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation cosh does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload

define noundef double @cosh_double(double noundef %a) {
entry:
  %1 = call double @llvm.cosh.f64(double %a)
  ret double %1
}
