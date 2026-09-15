; REQUIRES: amdgpu-registered-target
; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes='lto-pre-link<O1>' \
; RUN:   -amdgpu-enable-hipstdpar %s \
; RUN:   | opt -S -mtriple=amdgcn-amd-amdhsa -passes='lto<O1>' \
; RUN:       -amdgpu-enable-hipstdpar \
; RUN:   | FileCheck %s --implicit-check-not=llvm.sincos

define linkonce_odr double @mSin(double %x) #0 {
  %sin = call double @llvm.sin.f64(double %x)
  ret double %sin
}

define linkonce_odr double @mCos(double %x) #0 {
  %cos = call double @llvm.cos.f64(double %x)
  ret double %cos
}

define amdgpu_kernel void @test(double %x, ptr addrspace(1) %out) {
; CHECK-LABEL: define amdgpu_kernel void @test(
; CHECK:         [[SIN:%.*]] = {{.*}}call double @__hipstdpar_sin_f64(double %x)
; CHECK:         [[COS:%.*]] = {{.*}}call double @__hipstdpar_cos_f64(double %x)
; CHECK:         [[SUM:%.*]] = fadd double [[SIN]], [[COS]]
; CHECK:         store double [[SUM]], ptr addrspace(1) %out
;
entry:
  %sin = call double @mSin(double %x)
  %cos = call double @mCos(double %x)
  %sum = fadd double %sin, %cos
  store double %sum, ptr addrspace(1) %out
  ret void
}

declare double @llvm.sin.f64(double)
declare double @llvm.cos.f64(double)

attributes #0 = { alwaysinline }
