; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation dot2 does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload

define noundef double @dot_double2(<2 x double> noundef %a, <2 x double> noundef %b) {
entry:
  %dx.dot = call double @llvm.dx.dot2.v2f64(<2 x double> %a, <2 x double> %b)
  ret double %dx.dot
}
