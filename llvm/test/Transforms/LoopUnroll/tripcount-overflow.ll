; RUN: opt < %s -S -unroll-runtime -unroll-count=2 -loop-unroll | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -S -unroll-runtime -unroll-count=2 -loop-unroll -unroll-runtime-epilog=false | FileCheck %s -check-prefix=PROLOG
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

; This test case documents how runtime loop unrolling handles the case
; when the backedge-count is -1.

; If %N, the backedge-taken count, is -1 then %0 unsigned-overflows
; and is 0.  %xtraiter too is 0, signifying that the total trip-count
; is divisible by 2.  The prologue then branches to the unrolled loop
; and executes the 2^32 iterations there, in groups of 2.

; EPILOG: entry:

; EPILOG-NEXT: %0 = add i32 %N, 1
; EPILOG-NEXT: %xtraiter = and i32 %0, 1
; EPILOG-NEXT: %1 = icmp ult i32 %N, 1
; EPILOG-NEXT: br i1 %1, label %while.end.unr-lcssa, label %entry.new
; EPILOG: while.body:

; EPILOG: %lcmp.mod = icmp ne i32 %xtraiter, 0
; EPILOG-NEXT: br i1 %lcmp.mod, label %while.body.epil.preheader, label %while.end
; EPILOG: while.body.epil:

; PROLOG: entry:
; PROLOG-NEXT: %0 = add i32 %N, 1
; PROLOG-NEXT: %xtraiter = and i32 %0, 1
; PROLOG-NEXT: %lcmp.mod = icmp ne i32 %xtraiter, 0
; PROLOG-NEXT: br i1 %lcmp.mod, label %while.body.prol.preheader, label %while.body.prol.loopexit
; PROLOG: while.body.prol:

; PROLOG: %1 = icmp ult i32 %N, 1
; PROLOG-NEXT: br i1 %1, label %while.end, label %entry.new
; PROLOG: while.body:

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
