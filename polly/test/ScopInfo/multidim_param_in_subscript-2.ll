; RUN: opt %loadPolly -polly-scops -analyze \
; RUN:  -polly-precise-fold-accesses  < %s | FileCheck %s
;
;    void foo(long n, long m, float A[][n][m]) {
;      for (long i = 0; i < 100; i++)
;        for (long j = 0; j < n; j++)
;          for (long k = 0; k < m; k++)
;            A[i][j][k] += A[i][n - j - 1][m - k - 1];
;    }
;
; Verify that the parameter in the subscript expression is correctly
; recovered.
;
; CHECK: Assumed Context:
; CHECK-NEXT: [n, m] -> {  :  }
; CHECK: ReadAccess :=       [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT: [n, m] -> { Stmt_for_body6[i0, i1, i2] -> MemRef_A[i0, -1 + n - i1, -1 + m - i2] };
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(i64 %n, i64 %m, float* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc18, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc19, %for.inc18 ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %for.body, label %for.end20

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc15, %for.body
  %j.0 = phi i64 [ 0, %for.body ], [ %inc16, %for.inc15 ]
  %cmp2 = icmp slt i64 %j.0, %n
  br i1 %cmp2, label %for.body3, label %for.end17

for.body3:                                        ; preds = %for.cond1
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i64 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %cmp5 = icmp slt i64 %k.0, %m
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %sub = sub nsw i64 %m, %k.0
  %sub7 = add nsw i64 %sub, -1
  %sub8 = sub nsw i64 %n, %j.0
  %sub9 = add nsw i64 %sub8, -1
  %tmp = mul nuw i64 %n, %m
  %tmp1 = mul nsw i64 %i.0, %tmp
  %tmp2 = mul nsw i64 %sub9, %m
  %arrayidx.sum = add i64 %tmp1, %tmp2
  %arrayidx10.sum = add i64 %arrayidx.sum, %sub7
  %arrayidx11 = getelementptr inbounds float, float* %A, i64 %arrayidx10.sum
  %tmp3 = load float, float* %arrayidx11, align 4
  %tmp4 = mul nuw i64 %n, %m
  %tmp5 = mul nsw i64 %i.0, %tmp4
  %tmp6 = mul nsw i64 %j.0, %m
  %arrayidx12.sum = add i64 %tmp5, %tmp6
  %arrayidx13.sum = add i64 %arrayidx12.sum, %k.0
  %arrayidx14 = getelementptr inbounds float, float* %A, i64 %arrayidx13.sum
  %tmp7 = load float, float* %arrayidx14, align 4
  %add = fadd float %tmp7, %tmp3
  store float %add, float* %arrayidx14, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %inc = add nuw nsw i64 %k.0, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  br label %for.inc15

for.inc15:                                        ; preds = %for.end
  %inc16 = add nuw nsw i64 %j.0, 1
  br label %for.cond1

for.end17:                                        ; preds = %for.cond1
  br label %for.inc18

for.inc18:                                        ; preds = %for.end17
  %inc19 = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end20:                                        ; preds = %for.cond
  ret void
}
