; RUN: opt %loadNPMPolly '-passes=polly-opt-isl,print<polly-ast>' -disable-output < %s | FileCheck %s
;
;    void jacobi1d(long T, long N, float *A, float *B) {
;      long t, i, j;
;      for (t = 0; t < T; t++) {
;        for (i = 1; i < N - 1; i++)
;          B[i] = 0.33333 * (A[i - 1] + A[i] + A[i + 1]);
;        for (j = 1; j < N - 1; j++)
;          A[j] = 0.33333 * (B[i - 1] + B[i] + B[i + 1]);
;      }
;    }

; Verify that we do not tile bands that have just a single dimension.

; CHECK: for (int c0 = 0; c0 < T; c0 += 1) {
; CHECK:   for (int c1 = 0; c1 < N - 2; c1 += 1)
; CHECK:     Stmt_for_body3(c0, c1);
; CHECK:   for (int c1 = 0; c1 < N - 2; c1 += 1)
; CHECK:     Stmt_for_body15(c0, c1);
; CHECK: }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jacobi1d(i64 %T, i64 %N, ptr %A, ptr %B) {
entry:
  %tmp = add i64 %N, -1
  %tmp1 = icmp sgt i64 %tmp, 1
  %smax = select i1 %tmp1, i64 %tmp, i64 1
  br label %for.cond

for.cond:                                         ; preds = %for.inc30, %entry
  %t.0 = phi i64 [ 0, %entry ], [ %inc31, %for.inc30 ]
  %cmp = icmp slt i64 %t.0, %T
  br i1 %cmp, label %for.body, label %for.end32

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %i.0 = phi i64 [ 1, %for.body ], [ %inc, %for.inc ]
  %sub = add nsw i64 %N, -1
  %cmp2 = icmp slt i64 %i.0, %sub
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %sub4 = add nsw i64 %i.0, -1
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %sub4
  %tmp2 = load float, ptr %arrayidx, align 4
  %arrayidx5 = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp3 = load float, ptr %arrayidx5, align 4
  %add = fadd float %tmp2, %tmp3
  %add6 = add nuw nsw i64 %i.0, 1
  %arrayidx7 = getelementptr inbounds float, ptr %A, i64 %add6
  %tmp4 = load float, ptr %arrayidx7, align 4
  %add8 = fadd float %add, %tmp4
  %conv = fpext float %add8 to double
  %mul = fmul double %conv, 3.333300e-01
  %conv9 = fptrunc double %mul to float
  %arrayidx10 = getelementptr inbounds float, ptr %B, i64 %i.0
  store float %conv9, ptr %arrayidx10, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nuw nsw i64 %i.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.cond11

for.cond11:                                       ; preds = %for.inc27, %for.end
  %j.0 = phi i64 [ 1, %for.end ], [ %inc28, %for.inc27 ]
  %sub12 = add nsw i64 %N, -1
  %cmp13 = icmp slt i64 %j.0, %sub12
  br i1 %cmp13, label %for.body15, label %for.end29

for.body15:                                       ; preds = %for.cond11
  %sub16 = add nsw i64 %smax, -1
  %arrayidx17 = getelementptr inbounds float, ptr %B, i64 %sub16
  %tmp5 = load float, ptr %arrayidx17, align 4
  %arrayidx18 = getelementptr inbounds float, ptr %B, i64 %smax
  %tmp6 = load float, ptr %arrayidx18, align 4
  %add19 = fadd float %tmp5, %tmp6
  %add20 = add nsw i64 %smax, 1
  %arrayidx21 = getelementptr inbounds float, ptr %B, i64 %add20
  %tmp7 = load float, ptr %arrayidx21, align 4
  %add22 = fadd float %add19, %tmp7
  %conv23 = fpext float %add22 to double
  %mul24 = fmul double %conv23, 3.333300e-01
  %conv25 = fptrunc double %mul24 to float
  %arrayidx26 = getelementptr inbounds float, ptr %A, i64 %j.0
  store float %conv25, ptr %arrayidx26, align 4
  br label %for.inc27

for.inc27:                                        ; preds = %for.body15
  %inc28 = add nuw nsw i64 %j.0, 1
  br label %for.cond11

for.end29:                                        ; preds = %for.cond11
  br label %for.inc30

for.inc30:                                        ; preds = %for.end29
  %inc31 = add nuw nsw i64 %t.0, 1
  br label %for.cond

for.end32:                                        ; preds = %for.cond
  ret void
}
