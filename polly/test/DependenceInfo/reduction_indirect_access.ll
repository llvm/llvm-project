; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa '-passes=print<polly-dependences>' -polly-allow-nonaffine -disable-output < %s | FileCheck %s
;
; CHECK: Reduction dependences:
; CHECK:   [N] -> { Stmt_for_body[i0] -> Stmt_for_body[1 + i0] : 0 <= i0 <= -2 + N }
;
;    void f(double *restrict A, int *restrict INDICES, int N) {
;      for (int i = 0; i < N; i++)
;        A[INDICES[i]] += N;
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(ptr noalias %A, ptr noalias %INDICES, i32 %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %conv = sitofp i32 %N to double
  %arrayidx = getelementptr inbounds ptr, ptr %INDICES, i32 %i.0
  %tmp = load i32, ptr %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds ptr, ptr %A, i32 %tmp
  %tmp1 = load double, ptr %arrayidx1, align 8
  %add = fadd fast double %tmp1, %conv
  store double %add, double* %arrayidx1, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

