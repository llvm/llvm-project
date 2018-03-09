; RUN: opt -gvn-hoist -newgvn -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@GlobalVar = internal global float 1.000000e+00

; Check that we hoist load and scalar expressions in dominator.
; CHECK-LABEL: @dominatorHoisting
; CHECK: load
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK-NOT: load
; CHECK-NOT: fmul
; CHECK-NOT: fsub
define float @dominatorHoisting(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.end:                                          ; preds = %entry
  %p1 = phi float [ %mul4, %if.then ], [ 0.000000e+00, %entry ]
  %p2 = phi float [ %mul6, %if.then ], [ 0.000000e+00, %entry ]

  %x = fadd float %p1, %mul2
  %y = fadd float %p2, %mul
  %z = fadd float %x, %y
  ret float %z
}

; Check that we hoist load and scalar expressions in dominator.
; CHECK-LABEL: @domHoisting
; CHECK: load
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK: load
; CHECK: fsub
; CHECK: fmul
; CHECK-NOT: load
; CHECK-NOT: fmul
; CHECK-NOT: fsub
define float @domHoisting(float %d, float* %min, float* %max, float* %a) {
entry:
  %div = fdiv float 1.000000e+00, %d
  %0 = load float, float* %min, align 4
  %1 = load float, float* %a, align 4
  %sub = fsub float %0, %1
  %mul = fmul float %sub, %div
  %2 = load float, float* %max, align 4
  %sub1 = fsub float %2, %1
  %mul2 = fmul float %sub1, %div
  %cmp = fcmp oge float %div, 0.000000e+00
  br i1 %cmp, label %if.then, label %if.else

if.then:
  %3 = load float, float* %max, align 4
  %4 = load float, float* %a, align 4
  %sub3 = fsub float %3, %4
  %mul4 = fmul float %sub3, %div
  %5 = load float, float* %min, align 4
  %sub5 = fsub float %5, %4
  %mul6 = fmul float %sub5, %div
  br label %if.end

if.else:
  %6 = load float, float* %max, align 4
  %7 = load float, float* %a, align 4
  %sub9 = fsub float %6, %7
  %mul10 = fmul float %sub9, %div
  %8 = load float, float* %min, align 4
  %sub12 = fsub float %8, %7
  %mul13 = fmul float %sub12, %div
  br label %if.end

if.end:
  %p1 = phi float [ %mul4, %if.then ], [ %mul10, %if.else ]
  %p2 = phi float [ %mul6, %if.then ], [ %mul13, %if.else ]

  %x = fadd float %p1, %mul2
  %y = fadd float %p2, %mul
  %z = fadd float %x, %y
  ret float %z
}
