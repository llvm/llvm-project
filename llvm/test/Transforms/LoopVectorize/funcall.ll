; RUN: opt -S -passes=loop-vectorize -force-vector-width=2 -force-vector-interleave=1 < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; Make sure we can vectorize loops with functions to math library functions.
; They might read the rounding mode but we are only vectorizing loops that
; contain a limited set of function calls and none of them sets the rounding
; mode, so vectorizing them is safe.

; CHECK-LABEL: @test(
; CHECK: <2 x double>

define void @test(ptr %d, double %t) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %d, i64 %indvars.iv
  %0 = load double, ptr %arrayidx, align 8
  %1 = tail call double @llvm.pow.f64(double %0, double %t)
  store double %1, ptr %arrayidx, align 8
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body, label %for.end

for.end:
  ret void
}

declare double @llvm.pow.f64(double, double)
