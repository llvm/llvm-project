; Double precision operations the soft-f64 library has no emulation for are
; errors, not left as doubles.
; RUN: not opt -mtriple=air -passes=air-emulate-f64 -disable-output %s 2>&1 | FileCheck %s

; CHECK-DAG: error: {{.*}}in function rem {{.*}}air-emulate-f64: no emulation of inst:frem|ddd
; CHECK-DAG: error: {{.*}}in function transcendental {{.*}}air-emulate-f64: no emulation of llvm.exp.f64

target triple = "air"

declare double @llvm.exp.f64(double)

define void @rem(ptr addrspace(1) %p, double %x, double %y) "air-kernel" {
  %r = frem double %x, %y
  store double %r, ptr addrspace(1) %p
  ret void
}

define void @transcendental(ptr addrspace(1) %p, double %x) "air-kernel" {
  %e = call double @llvm.exp.f64(double %x)
  store double %e, ptr addrspace(1) %p
  ret void
}
