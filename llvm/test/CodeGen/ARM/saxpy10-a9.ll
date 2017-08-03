; RUN: llc < %s -march=arm -mtriple=thumbv7-apple-ios7.0.0 -float-abi=hard -mcpu=cortex-a9 -misched-postra -enable-misched -pre-RA-sched=source -scheditins=false | FileCheck %s
;
; Test MI-Sched suppory latency based stalls on in in-order pipeline
; using the new machine model.

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"

; Don't be too strict with the top of the schedule, but most of it
; should be nicely pipelined.
;
; CHECK: saxpy10:
; CHECK: vldr
; CHECK: vldr
; CHECK: vldr
; CHECK: vldr
; CHECK: vldr
; CHECK-NEXT: vldr
; CHECK-NEXT: vmul
; CHECK-NEXT: vadd
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vldr
; CHECK-NEXT: vadd
; CHECK-NEXT: vadd
; CHECK-NEXT: vmul
; CHECK-NEXT: vldr
; CHECK-NEXT: vadd
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vmul
; CHECK-NEXT: vldr
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vmul
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vmul
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vmul
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vmul
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vmul
; CHECK-NEXT: vadd
; CHECK-NEXT: vldr
; CHECK-NEXT: vadd
; CHECK-NEXT: vadd
; CHECK-NEXT: vadd
; CHECK-NEXT: vmov
; CHECK-NEXT: bx
;
; This accumulates a sum rather than storing each result.
define float @saxpy10(float* nocapture readonly %data1, float* nocapture readonly %data2, float %a) {
entry:
  %0 = load float, float* %data1, align 4
  %mul = fmul float %0, %a
  %1 = load float, float* %data2, align 4
  %add = fadd float %mul, %1
  %add2 = fadd float %add, 0.000000e+00
  %arrayidx.1 = getelementptr inbounds float, float* %data1, i32 1
  %2 = load float, float* %arrayidx.1, align 4
  %mul.1 = fmul float %2, %a
  %arrayidx1.1 = getelementptr inbounds float, float* %data2, i32 1
  %3 = load float, float* %arrayidx1.1, align 4
  %add.1 = fadd float %mul.1, %3
  %add2.1 = fadd float %add2, %add.1
  %arrayidx.2 = getelementptr inbounds float, float* %data1, i32 2
  %4 = load float, float* %arrayidx.2, align 4
  %mul.2 = fmul float %4, %a
  %arrayidx1.2 = getelementptr inbounds float, float* %data2, i32 2
  %5 = load float, float* %arrayidx1.2, align 4
  %add.2 = fadd float %mul.2, %5
  %add2.2 = fadd float %add2.1, %add.2
  %arrayidx.3 = getelementptr inbounds float, float* %data1, i32 3
  %6 = load float, float* %arrayidx.3, align 4
  %mul.3 = fmul float %6, %a
  %arrayidx1.3 = getelementptr inbounds float, float* %data2, i32 3
  %7 = load float, float* %arrayidx1.3, align 4
  %add.3 = fadd float %mul.3, %7
  %add2.3 = fadd float %add2.2, %add.3
  %arrayidx.4 = getelementptr inbounds float, float* %data1, i32 4
  %8 = load float, float* %arrayidx.4, align 4
  %mul.4 = fmul float %8, %a
  %arrayidx1.4 = getelementptr inbounds float, float* %data2, i32 4
  %9 = load float, float* %arrayidx1.4, align 4
  %add.4 = fadd float %mul.4, %9
  %add2.4 = fadd float %add2.3, %add.4
  %arrayidx.5 = getelementptr inbounds float, float* %data1, i32 5
  %10 = load float, float* %arrayidx.5, align 4
  %mul.5 = fmul float %10, %a
  %arrayidx1.5 = getelementptr inbounds float, float* %data2, i32 5
  %11 = load float, float* %arrayidx1.5, align 4
  %add.5 = fadd float %mul.5, %11
  %add2.5 = fadd float %add2.4, %add.5
  %arrayidx.6 = getelementptr inbounds float, float* %data1, i32 6
  %12 = load float, float* %arrayidx.6, align 4
  %mul.6 = fmul float %12, %a
  %arrayidx1.6 = getelementptr inbounds float, float* %data2, i32 6
  %13 = load float, float* %arrayidx1.6, align 4
  %add.6 = fadd float %mul.6, %13
  %add2.6 = fadd float %add2.5, %add.6
  %arrayidx.7 = getelementptr inbounds float, float* %data1, i32 7
  %14 = load float, float* %arrayidx.7, align 4
  %mul.7 = fmul float %14, %a
  %arrayidx1.7 = getelementptr inbounds float, float* %data2, i32 7
  %15 = load float, float* %arrayidx1.7, align 4
  %add.7 = fadd float %mul.7, %15
  %add2.7 = fadd float %add2.6, %add.7
  %arrayidx.8 = getelementptr inbounds float, float* %data1, i32 8
  %16 = load float, float* %arrayidx.8, align 4
  %mul.8 = fmul float %16, %a
  %arrayidx1.8 = getelementptr inbounds float, float* %data2, i32 8
  %17 = load float, float* %arrayidx1.8, align 4
  %add.8 = fadd float %mul.8, %17
  %add2.8 = fadd float %add2.7, %add.8
  %arrayidx.9 = getelementptr inbounds float, float* %data1, i32 9
  %18 = load float, float* %arrayidx.9, align 4
  %mul.9 = fmul float %18, %a
  %arrayidx1.9 = getelementptr inbounds float, float* %data2, i32 9
  %19 = load float, float* %arrayidx1.9, align 4
  %add.9 = fadd float %mul.9, %19
  %add2.9 = fadd float %add2.8, %add.9
  ret float %add2.9
}
