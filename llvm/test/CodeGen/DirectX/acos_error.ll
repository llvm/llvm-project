; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation acos does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload

define noundef double @acos_double(double noundef %a) {
entry:
  %1 = call double @llvm.acos.f64(double %a)
  ret double %1
}
