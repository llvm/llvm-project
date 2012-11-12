; RUN: llc < %s -mtriple=x86_64-linux -mattr=+sse2 -mcpu=nehalem | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-win32 -mattr=+sse2 -mcpu=nehalem | FileCheck %s

define double @t1(float* nocapture %x) nounwind readonly ssp {
entry:
; CHECK: t1:
; CHECK: movss ([[A0:%rdi|%rcx]]), %xmm0
; CHECK: cvtss2sd %xmm0, %xmm0

  %0 = load float* %x, align 4
  %1 = fpext float %0 to double
  ret double %1
}

define float @t2(double* nocapture %x) nounwind readonly ssp optsize {
entry:
; CHECK: t2:
; CHECK: cvtsd2ss ([[A0]]), %xmm0
  %0 = load double* %x, align 8
  %1 = fptrunc double %0 to float
  ret float %1
}

define float @squirtf(float* %x) nounwind {
entry:
; CHECK: squirtf:
; CHECK: movss ([[A0]]), %xmm0
; CHECK: sqrtss %xmm0, %xmm0
  %z = load float* %x
  %t = call float @llvm.sqrt.f32(float %z)
  ret float %t
}

define double @squirt(double* %x) nounwind {
entry:
; CHECK: squirt:
; CHECK: sqrtsd ([[A0]]), %xmm0
  %z = load double* %x
  %t = call double @llvm.sqrt.f64(double %z)
  ret double %t
}

define float @squirtf_size(float* %x) nounwind optsize {
entry:
; CHECK: squirtf_size:
; CHECK: sqrtss ([[A0]]), %xmm0
  %z = load float* %x
  %t = call float @llvm.sqrt.f32(float %z)
  ret float %t
}

define double @squirt_size(double* %x) nounwind optsize {
entry:
; CHECK: squirt_size:
; CHECK: sqrtsd ([[A0]]), %xmm0
  %z = load double* %x
  %t = call double @llvm.sqrt.f64(double %z)
  ret double %t
}

declare float @llvm.sqrt.f32(float)
declare double @llvm.sqrt.f64(double)
