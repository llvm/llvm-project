; RUN: opt < %s -S -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 -loop-unroll | FileCheck %s -check-prefix=EPILOG
; RUN: opt < %s -S -mtriple=powerpc64-unknown-linux-gnu -mcpu=a2 -loop-unroll -unroll-runtime-epilog=false | FileCheck %s -check-prefix=PROLOG
define void @unroll_opt_for_size() nounwind optsize {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %inc = add i32 %iv, 1
  %exitcnd = icmp uge i32 %inc, 1024
  br i1 %exitcnd, label %exit, label %loop

exit:
  ret void
}

; EPILOG-LABEL: @unroll_opt_for_size
; EPILOG:      add
; EPILOG-NEXT: add
; EPILOG-NEXT: add
; EPILOG: icmp

; PROLOG-LABEL: @unroll_opt_for_size
; PROLOG:      add
; PROLOG-NEXT: add
; PROLOG-NEXT: add
; PROLOG: icmp

define i32 @test(i32* nocapture %a, i32 %n) nounwind uwtable readonly {
entry:
  %cmp1 = icmp eq i32 %n, 0
  br i1 %cmp1, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %sum.02 = phi i32 [ %add, %for.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %add = add nsw i32 %0, %sum.02
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %for.body ]
  ret i32 %sum.0.lcssa
}

; EPILOG-LABEL: @test
; EPILOG: for.body:
; EPILOG: br i1 %niter.ncmp.7, label %for.end.loopexit{{.*}}, label %for.body
; EPILOG: for.body.epil{{.*}}:

; PROLOG-LABEL: @test
; PROLOG: for.body.prol{{.*}}:
; PROLOG: for.body:
; PROLOG: br i1 %exitcond.7, label %for.end.loopexit{{.*}}, label %for.body

