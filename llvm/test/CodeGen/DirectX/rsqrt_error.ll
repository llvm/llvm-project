; RUN: not opt -S -dxil-op-lower %s 2>&1 | FileCheck %s

; DXIL operation rsqrt does not support double overload type
; CHECK: LLVM ERROR: Invalid Overload Type

; Function Attrs: noinline nounwind optnone
define noundef double @rsqrt_double(double noundef %a) #0 {
entry:
  %a.addr = alloca double, align 8
  store double %a, ptr %a.addr, align 8
  %0 = load double, ptr %a.addr, align 8
  %dx.rsqrt = call double @llvm.dx.rsqrt.f64(double %0)
  ret double %dx.rsqrt
}
