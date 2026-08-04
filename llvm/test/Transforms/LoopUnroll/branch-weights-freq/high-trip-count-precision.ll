; RUN: opt -S -passes=loop-unroll < %s | FileCheck %s
;
; Test that loop unroll does not crash with an assertion failure when the
; original loop probability is very close to 1 (high trip count), which causes
; FreqDesired to be very large (e.g., ~10^7).  The assertion was using an
; absolute tolerance (1e-6) to verify computed frequency accuracy, which is
; insufficient for large frequency values due to floating-point precision
; limits.  The fix uses a relative tolerance instead.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @fn2(
; CHECK: vector.body:
; CHECK: br i1 %{{.*}}, label %scalar.ph, label %vector.body
; CHECK: scalar.ph:
; CHECK: ret void
define fastcc void @fn2() #0 {
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %index = phi i32 [ 0, %entry ], [ %index.next, %vector.body ]
  %index.next = add i32 %index, 8
  %0 = icmp eq i32 %index.next, 80405912
  br i1 %0, label %scalar.ph, label %vector.body, !prof !0

scalar.ph:                                        ; preds = %vector.body
  ret void
}

attributes #0 = { "target-cpu"="x86-64" }

; Branch weights indicating very high trip count (~10 million iterations).
!0 = !{!"branch_weights", i32 1, i32 10050738}
