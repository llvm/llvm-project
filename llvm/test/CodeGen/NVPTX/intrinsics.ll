; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s

; CHECK-LABEL test_fabsf(
define float @test_fabsf(float %f) {
; CHECK: abs.f32
  %x = call float @llvm.fabs.f32(float %f)
  ret float %x
}

; CHECK-LABEL: test_fabs(
define double @test_fabs(double %d) {
; CHECK: abs.f64
  %x = call double @llvm.fabs.f64(double %d)
  ret double %x
}

; CHECK-LABEL: test_nvvm_sqrt(
define float @test_nvvm_sqrt(float %a) {
; CHECK: sqrt.rn.f32
  %val = call float @llvm.nvvm.sqrt.f(float %a)
  ret float %val
}

declare float @llvm.fabs.f32(float)
declare double @llvm.fabs.f64(double)
declare float @llvm.nvvm.sqrt.f(float)
