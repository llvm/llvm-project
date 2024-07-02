; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation dot4 does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload

define noundef double @dot_double4(<4 x double> noundef %a, <4 x double> noundef %b) {
entry:
  %dx.dot = call double @llvm.dx.dot4.v4f64(<4 x double> %a, <4 x double> %b)
  ret double %dx.dot
}
