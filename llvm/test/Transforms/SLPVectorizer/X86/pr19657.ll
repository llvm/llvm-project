; RUN: opt < %s -basicaa -slp-vectorizer -S -mcpu=corei7-avx | FileCheck %s
; RUN: opt < %s -basicaa -slp-vectorizer -slp-max-reg-size=128 -S -mcpu=corei7-avx | FileCheck %s --check-prefix=V128

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @foo(
; CHECK: load <4 x double>
; CHECK: fadd <4 x double>
; CHECK: fadd <4 x double>
; CHECK: store <4 x double>

; V128-LABEL: @foo(
; V128: load <2 x double>
; V128: fadd <2 x double>
; V128: fadd <2 x double>
; V128: store <2 x double>
; V128: load <2 x double>
; V128: fadd <2 x double>
; V128: fadd <2 x double>
; V128: store <2 x double>

define void @foo(double* %x) {
  %1 = load double, double* %x, align 8
  %2 = fadd double %1, %1
  %3 = fadd double %2, %1
  store double %3, double* %x, align 8
  %4 = getelementptr inbounds double, double* %x, i64 1
  %5 = load double, double* %4, align 8
  %6 = fadd double %5, %5
  %7 = fadd double %6, %5
  store double %7, double* %4, align 8
  %8 = getelementptr inbounds double, double* %x, i64 2
  %9 = load double, double* %8, align 8
  %10 = fadd double %9, %9
  %11 = fadd double %10, %9
  store double %11, double* %8, align 8
  %12 = getelementptr inbounds double, double* %x, i64 3
  %13 = load double, double* %12, align 8
  %14 = fadd double %13, %13
  %15 = fadd double %14, %13
  store double %15, double* %12, align 8
  ret void
}

