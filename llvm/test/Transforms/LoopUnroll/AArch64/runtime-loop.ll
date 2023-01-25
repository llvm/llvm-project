; RUN: opt < %s -S -passes=loop-unroll -mtriple aarch64 -mcpu=cortex-a57 -unroll-runtime-epilog=true  | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -S -passes=loop-unroll -mtriple aarch64 -mcpu=cortex-a57 -unroll-runtime-epilog=false | FileCheck %s -check-prefix=PROLOG
; RUN: opt < %s -S -passes=loop-unroll -mtriple aarch64 -mcpu=cortex-r82 -unroll-runtime-epilog=true  | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -S -passes=loop-unroll -mtriple aarch64 -mcpu=cortex-r82 -unroll-runtime-epilog=false | FileCheck %s -check-prefix=PROLOG

; Tests for unrolling loops with run-time trip counts

; EPILOG:  %xtraiter = and i32 %n
; EPILOG:  for.body:
; EPILOG:  %lcmp.mod = icmp ne i32 %xtraiter, 0
; EPILOG:  br i1 %lcmp.mod, label %for.body.epil.preheader, label %for.end.loopexit
; EPILOG:  for.body.epil:

; PROLOG:  %xtraiter = and i32 %n
; PROLOG:  %lcmp.mod = icmp ne i32 %xtraiter, 0
; PROLOG:  br i1 %lcmp.mod, label %for.body.prol.preheader, label %for.body.prol.loopexit
; PROLOG:  for.body.prol:
; PROLOG:  for.body:

define i32 @test(ptr nocapture %a, i32 %n) nounwind uwtable readonly {
entry:
  %cmp1 = icmp eq i32 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa
}


