; RUN: llc -mtriple=air %s -o - | FileCheck %s

; CHECK-LABEL: define void @kernel
; CHECK: sitofp i64 %{{.*}} to float
; CHECK: call float @air.fast_sqrt.f32(float
; CHECK: call float @air.fast_log.f32(float
; CHECK: fmul float
; CHECK: call float @air.fast_exp.f32(float
; CHECK-NOT: fptrunc
; CHECK: store float
; CHECK-NOT: sitofp i64 %{{.*}} to double
; CHECK-NOT: call double
; CHECK-NOT: fmul double

define void @kernel(ptr addrspace(1) %out, i64 %n) {
entry:
  %d = sitofp i64 %n to double
  %s = call double @llvm.sqrt.f64(double %d)
  %l = call double @llvm.log.f64(double %s)
  %m = fmul double %l, 1.000000e+01
  %e = call double @llvm.exp.f64(double %m)
  %f = fptrunc double %e to float
  store float %f, ptr addrspace(1) %out, align 4
  ret void
}

declare double @llvm.sqrt.f64(double)
declare double @llvm.log.f64(double)
declare double @llvm.exp.f64(double)
