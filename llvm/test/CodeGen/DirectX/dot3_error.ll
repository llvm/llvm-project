; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation dot3 does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload

define noundef double @dot_double3(<3 x double> noundef %a, <3 x double> noundef %b) {
entry:
  %dx.dot = call double @llvm.dx.dot3.v3f64(<3 x double> %a, <3 x double> %b)
  ret double %dx.dot
}
