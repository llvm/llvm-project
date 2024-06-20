; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; This test is expected to fail with the following error
; CHECK: LLVM ERROR: Invalid Overload Type

define noundef double @round_double(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %elt.roundeven = call double @llvm.roundeven.f64(double %0)
  ret double %elt.roundeven
}
