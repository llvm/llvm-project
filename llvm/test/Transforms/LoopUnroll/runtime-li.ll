; RUN: opt -S -passes=loop-unroll -unroll-runtime -unroll-count=2 -verify-loop-info -pass-remarks=loop-unroll < %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Verify that runtime-unrolling a top-level loop that has nested loops does not
; make the unroller produce invalid loop-info.
; CHECK: remark: {{.*}}: unrolled loop by a factor of 2 with run-time trip count
; CHECK: @widget
; CHECK: ret void
define void @widget(double* %arg, double* %arg1, double* %p, i64* %q1, i64* %q2, i1 %c) local_unnamed_addr {
entry:
  br label %header.outer

header.outer:                                     ; preds = %latch.outer, %entry
  %tmp = phi double* [ %tmp8, %latch.outer ], [ %arg, %entry ]
  br label %header.inner

header.inner:                                     ; preds = %latch.inner, %header.outer
  %tmp5 = load i64, i64* %q1, align 8
  %tmp6 = icmp eq double* %p, %arg
  br i1 %c, label %exiting.inner, label %latch.outer

exiting.inner:                                     ; preds = %latch.inner, %header.outer
  br i1 %c, label %latch.inner, label %latch.outer

latch.inner:                                      ; preds = %header.inner
  store i64 %tmp5, i64* %q2, align 8
  br label %header.inner

latch.outer:                                      ; preds = %header.inner
  store double 0.0, double* %p, align 8
  %tmp8 = getelementptr inbounds double, double* %tmp, i64 1
  %tmp9 = icmp eq double* %tmp8, %arg1
  br i1 %tmp9, label %exit, label %header.outer

exit:                                             ; preds = %latch.outer
  ret void
}
