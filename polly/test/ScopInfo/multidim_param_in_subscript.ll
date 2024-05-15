; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
;
;
;    void foo(long n, float A[][n]) {
;      for (long i = 0; i < 100; i++)
;        for (long j = 0; j < n; j++)
;          A[i][j] += A[i][n - j - 1];
;    }
;
; Verify that the parameter in the subscript expression is correctly
; recovered.
;
; CHECK: Assumed Context:
; CHECK-NEXT: [n] -> {  :  }
;
; CHECK: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:   [n] -> { Stmt_for_body3[i0, i1] -> MemRef_A[i0, -1 + n - i1] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %n, ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc8, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc9, %for.inc8 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %for.body, label %for.end10

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i64 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp2 = icmp slt i64 %j.0, %n
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %sub = sub nsw i64 %n, %j.0
  %sub4 = add nsw i64 %sub, -1
  %tmp = mul nsw i64 %i.0, %n
  %arrayidx.sum = add i64 %tmp, %sub4
  %arrayidx5 = getelementptr inbounds float, ptr %A, i64 %arrayidx.sum
  %tmp1 = load float, ptr %arrayidx5, align 4
  %tmp2 = mul nsw i64 %i.0, %n
  %arrayidx6.sum = add i64 %tmp2, %j.0
  %arrayidx7 = getelementptr inbounds float, ptr %A, i64 %arrayidx6.sum
  %tmp3 = load float, ptr %arrayidx7, align 4
  %add = fadd float %tmp3, %tmp1
  store float %add, ptr %arrayidx7, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nuw nsw i64 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc8

for.inc8:                                         ; preds = %for.end
  %inc9 = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end10:                                        ; preds = %for.cond
  ret void
}
