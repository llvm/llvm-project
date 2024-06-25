; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation tan does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload

define noundef double @tan_double(double noundef %a) #0 {
entry:
  %1 = call double @llvm.tan.f64(double %a)
  ret double %1
}
