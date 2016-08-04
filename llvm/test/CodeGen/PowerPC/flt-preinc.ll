; RUN: llc -verify-machineinstrs -mcpu=ppc64 < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind readonly
define float @tf(float* nocapture readonly %i, i32 signext %o) #0 {
entry:
  %idx.ext = sext i32 %o to i64
  %add.ptr = getelementptr inbounds float, float* %i, i64 %idx.ext
  %0 = load float, float* %add.ptr, align 4
  %add.ptr.sum = add nsw i64 %idx.ext, 1
  %add.ptr3 = getelementptr inbounds float, float* %i, i64 %add.ptr.sum
  %1 = load float, float* %add.ptr3, align 4
  %add = fadd float %0, %1
  ret float %add

; CHECK-LABEL: @tf
; CHECK: lfsux
; CHECK: blr
}

; Function Attrs: nounwind readonly
define double @td(double* nocapture readonly %i, i32 signext %o) #0 {
entry:
  %idx.ext = sext i32 %o to i64
  %add.ptr = getelementptr inbounds double, double* %i, i64 %idx.ext
  %0 = load double, double* %add.ptr, align 8
  %add.ptr.sum = add nsw i64 %idx.ext, 1
  %add.ptr3 = getelementptr inbounds double, double* %i, i64 %add.ptr.sum
  %1 = load double, double* %add.ptr3, align 8
  %add = fadd double %0, %1
  ret double %add

; CHECK-LABEL: @td
; CHECK: lfdux
; CHECK: blr
}

attributes #0 = { nounwind readonly }

