; RUN: opt < %s -S -unroll-runtime -unroll-count=2 -loop-unroll | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; This test case documents how runtime loop unrolling handles the case
; when the backedge-count is -1.

; If %N, the backedge-taken count, is -1 then %0 unsigned-overflows
; and is 0.  %xtraiter too is 0, signifying that the total trip-count
; is divisible by 2.  The prologue then branches to the unrolled loop
; and executes the 2^32 iterations there, in groups of 2.


; CHECK: entry:
; CHECK-NEXT: %0 = add i32 %N, 1
; CHECK-NEXT: %xtraiter = and i32 %0, 1
; CHECK-NEXT: %lcmp.mod = icmp ne i32 %xtraiter, %0
; CHECK-NEXT: br i1 %lcmp.mod, label %entry.new, label %while.end.unr-lcssa

; CHECK: while.body.epil:
; CHECK: br label %while.end.epilog-lcssa

; CHECK: while.end.epilog-lcssa:

; Function Attrs: nounwind readnone ssp uwtable
define i32 @foo(i32 %N) {
entry:
  br label %while.body

while.body:                                       ; preds = %while.body, %entry
  %i = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  %cmp = icmp eq i32 %i, %N
  %inc = add i32 %i, 1
  br i1 %cmp, label %while.end, label %while.body

while.end:                                        ; preds = %while.body
  ret i32 %i
}
