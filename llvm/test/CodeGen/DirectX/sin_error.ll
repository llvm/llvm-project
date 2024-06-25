; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation sin does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload

define noundef double @sin_double(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %1 = call double @llvm.sin.f64(double %0)
  ret double %1
}

