; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s
;
;    void f(char *A, short N) {
;      for (short i = 0; i < N; i++)
;        A[(char)(N)]++;
;    }
;
; FIXME: We should the truncate precisely... or just make it a separate parameter.
; CHECK:       Assumed Context:
; CHECK-NEXT:  [N] -> {  :  }
; CHECK-NEXT:  Invalid Context:
; CHECK-NEXT:  [N] -> { : N >= 128 }
;
; CHECK:         ReadAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:        [N] -> { Stmt_for_body[i0] -> MemRef_A[N] };
; CHECK-NEXT:    MustWriteAccess :=	[Reduction Type: +] [Scalar: 0]
; CHECK-NEXT:        [N] -> { Stmt_for_body[i0] -> MemRef_A[N] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i8* %A, i16 signext %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i16 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i16 %indvars.iv, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %idxprom = trunc i16 %N to i8
  %arrayidx = getelementptr inbounds i8, i8* %A, i8 %idxprom
  %tmp1 = load i8, i8* %arrayidx, align 1
  %inc = add i8 %tmp1, 1
  store i8 %inc, i8* %arrayidx, align 1
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i16 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
