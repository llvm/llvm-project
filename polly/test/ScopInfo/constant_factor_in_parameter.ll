; RUN: opt %loadPolly -analyze -polly-scops < %s | FileCheck %s
; RUN: opt %loadPolly -analyze -polly-function-scops < %s | FileCheck %s
;
; Check that the constant part of the N * M * 4 expression is not part of the
; parameter but explicit in the access function. This can avoid existentially
; quantified variables, e.g., when computing the stride.
;
; CHECK: p1: (%N * %M)
; CHECK: [N, p_1] -> { Stmt_for_body[i0] -> MemRef_A[4p_1 + i0] };
;
;    void f(int *A, int N, int M) {
;      for (int i = 0; i < N; i++)
;        A[i + N * M * 4] = i;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32* %A, i32 %N, i32 %M) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %mul = mul nsw i32 %N, %M
  %mul2 = mul nsw i32 %mul, 4
  %tmp2 = sext i32 %mul2 to i64
  %tmp3 = add nsw i64 %indvars.iv, %tmp2
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %tmp3
  %tmp4 = trunc i64 %indvars.iv to i32
  store i32 %tmp4, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
