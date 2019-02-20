; RUN: llc -mtriple=armv7 < %s | FileCheck %s
; RUN: llc -mtriple=thumbv7 < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "armv7--"

define double @f(double %a) {
  ; CHECK-LABEL: f:
  ; CHECK: bfc r1, #31, #1
  ; CHECK-NEXT: bx lr
  %x = call double @llvm.fabs.f64(double %a) readnone
  ret double %x
}

define float @g(float %a) {
  ; CHECK-LABEL: g:
  ; CHECK: bic r0, r0, #-2147483648
  ; CHECK-NEXT: bx lr
  %x = call float @llvm.fabs.f32(float %a) readnone
  ret float %x
}

define double @h(double %a) {
  ; CHECK-LABEL: h:
  ; CHECK: eor r1, r1, #-2147483648
  ; CHECK-NEXT: bx lr
  %x = fsub nsz double -0.0, %a
  ret double %x
}

define float @i(float %a) {
  ; CHECK-LABEL: i:
  ; CHECK: eor r0, r0, #-2147483648
  ; CHECK-NEXT: bx lr
  %x = fsub nsz float -0.0, %a
  ret float %x
}

declare double @llvm.fabs.f64(double) readnone
declare float @llvm.fabs.f32(float) readnone
