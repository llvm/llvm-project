; RUN: llc < %s -mcpu=cortex-a57 -mattr=+neon -fp-contract=fast -regalloc=pbqp -pbqp-coalescing | FileCheck %s --check-prefix CHECK --check-prefix CHECK-EVEN
; RUN: llc < %s -mcpu=cortex-a57 -mattr=+neon -fp-contract=fast -regalloc=pbqp -pbqp-coalescing | FileCheck %s --check-prefix CHECK --check-prefix CHECK-ODD
;
; Test PBQP is able to fulfill the accumulator chaining constraint.
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64"

; CHECK-LABEL: fir
; CHECK-EVEN: fmadd {{d[0-9]*[02468]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[02468]}}
; CHECK-EVEN: fmadd {{d[0-9]*[02468]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[02468]}}
; CHECK-EVEN: fmadd {{d[0-9]*[02468]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[02468]}}
; CHECK-EVEN: fmadd {{d[0-9]*[02468]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[02468]}}
; CHECK-EVEN: fmadd {{d[0-9]*[02468]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[02468]}}
; CHECK-EVEN: fmadd {{d[0-9]*[02468]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[02468]}}
; CHECK-EVEN: fmadd {{d[0-9]*[02468]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[02468]}}
; CHECK-ODD: fmadd {{d[0-9]*[13579]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[13579]}}
; CHECK-ODD: fmadd {{d[0-9]*[13579]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[13579]}}
; CHECK-ODD: fmadd {{d[0-9]*[13579]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[13579]}}
; CHECK-ODD: fmadd {{d[0-9]*[13579]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[13579]}}
; CHECK-ODD: fmadd {{d[0-9]*[13579]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[13579]}}
; CHECK-ODD: fmadd {{d[0-9]*[13579]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[13579]}}
; CHECK-ODD: fmadd {{d[0-9]*[13579]}}, {{d[0-9]*}}, {{d[0-9]*}}, {{d[0-9]*[13579]}}
define void @fir(ptr nocapture %rx, ptr nocapture %ry, ptr nocapture %c, ptr nocapture %x, ptr nocapture %y) {
entry:
  %0 = load double, ptr %c, align 8
  %1 = load double, ptr %x, align 8
  %mul = fmul fast double %1, %0
  %2 = load double, ptr %y, align 8
  %mul7 = fmul fast double %2, %0
  %arrayidx.1 = getelementptr inbounds double, ptr %c, i64 1
  %3 = load double, ptr %arrayidx.1, align 8
  %arrayidx2.1 = getelementptr inbounds double, ptr %x, i64 1
  %4 = load double, ptr %arrayidx2.1, align 8
  %mul.1 = fmul fast double %4, %3
  %add.1 = fadd fast double %mul.1, %mul
  %arrayidx6.1 = getelementptr inbounds double, ptr %y, i64 1
  %5 = load double, ptr %arrayidx6.1, align 8
  %mul7.1 = fmul fast double %5, %3
  %add8.1 = fadd fast double %mul7.1, %mul7
  %arrayidx.2 = getelementptr inbounds double, ptr %c, i64 2
  %6 = load double, ptr %arrayidx.2, align 8
  %arrayidx2.2 = getelementptr inbounds double, ptr %x, i64 2
  %7 = load double, ptr %arrayidx2.2, align 8
  %mul.2 = fmul fast double %7, %6
  %add.2 = fadd fast double %mul.2, %add.1
  %arrayidx6.2 = getelementptr inbounds double, ptr %y, i64 2
  %8 = load double, ptr %arrayidx6.2, align 8
  %mul7.2 = fmul fast double %8, %6
  %add8.2 = fadd fast double %mul7.2, %add8.1
  %arrayidx.3 = getelementptr inbounds double, ptr %c, i64 3
  %9 = load double, ptr %arrayidx.3, align 8
  %arrayidx2.3 = getelementptr inbounds double, ptr %x, i64 3
  %10 = load double, ptr %arrayidx2.3, align 8
  %mul.3 = fmul fast double %10, %9
  %add.3 = fadd fast double %mul.3, %add.2
  %arrayidx6.3 = getelementptr inbounds double, ptr %y, i64 3
  %11 = load double, ptr %arrayidx6.3, align 8
  %mul7.3 = fmul fast double %11, %9
  %add8.3 = fadd fast double %mul7.3, %add8.2
  %arrayidx.4 = getelementptr inbounds double, ptr %c, i64 4
  %12 = load double, ptr %arrayidx.4, align 8
  %arrayidx2.4 = getelementptr inbounds double, ptr %x, i64 4
  %13 = load double, ptr %arrayidx2.4, align 8
  %mul.4 = fmul fast double %13, %12
  %add.4 = fadd fast double %mul.4, %add.3
  %arrayidx6.4 = getelementptr inbounds double, ptr %y, i64 4
  %14 = load double, ptr %arrayidx6.4, align 8
  %mul7.4 = fmul fast double %14, %12
  %add8.4 = fadd fast double %mul7.4, %add8.3
  %arrayidx.5 = getelementptr inbounds double, ptr %c, i64 5
  %15 = load double, ptr %arrayidx.5, align 8
  %arrayidx2.5 = getelementptr inbounds double, ptr %x, i64 5
  %16 = load double, ptr %arrayidx2.5, align 8
  %mul.5 = fmul fast double %16, %15
  %add.5 = fadd fast double %mul.5, %add.4
  %arrayidx6.5 = getelementptr inbounds double, ptr %y, i64 5
  %17 = load double, ptr %arrayidx6.5, align 8
  %mul7.5 = fmul fast double %17, %15
  %add8.5 = fadd fast double %mul7.5, %add8.4
  %arrayidx.6 = getelementptr inbounds double, ptr %c, i64 6
  %18 = load double, ptr %arrayidx.6, align 8
  %arrayidx2.6 = getelementptr inbounds double, ptr %x, i64 6
  %19 = load double, ptr %arrayidx2.6, align 8
  %mul.6 = fmul fast double %19, %18
  %add.6 = fadd fast double %mul.6, %add.5
  %arrayidx6.6 = getelementptr inbounds double, ptr %y, i64 6
  %20 = load double, ptr %arrayidx6.6, align 8
  %mul7.6 = fmul fast double %20, %18
  %add8.6 = fadd fast double %mul7.6, %add8.5
  %arrayidx.7 = getelementptr inbounds double, ptr %c, i64 7
  %21 = load double, ptr %arrayidx.7, align 8
  %arrayidx2.7 = getelementptr inbounds double, ptr %x, i64 7
  %22 = load double, ptr %arrayidx2.7, align 8
  %mul.7 = fmul fast double %22, %21
  %add.7 = fadd fast double %mul.7, %add.6
  %arrayidx6.7 = getelementptr inbounds double, ptr %y, i64 7
  %23 = load double, ptr %arrayidx6.7, align 8
  %mul7.7 = fmul fast double %23, %21
  %add8.7 = fadd fast double %mul7.7, %add8.6
  store double %add.7, ptr %rx, align 8
  store double %add8.7, ptr %ry, align 8
  ret void
}

