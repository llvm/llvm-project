; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation exp2 does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload

define noundef double @exp2_double(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %elt.exp2 = call double @llvm.exp2.f64(double %0)
  ret double %elt.exp2
}
