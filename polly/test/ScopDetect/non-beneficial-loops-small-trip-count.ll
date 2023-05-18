; RUN: opt %loadPolly -polly-process-unprofitable=false -polly-print-detect -disable-output < %s | FileCheck %s
;
; CHECK-NOT: Valid
;
; Do not consider this a SCoP as we do not perform any optimizations for
; loops with a small trip count.
;
;    void f(int *A) {
;      for (int i = 0; i < 4; i++)
;        for (int j = 0; j < 4; j++)
;          for (int k = 0; k < 4; k++)
;            for (int l = 0; l < 4; l++)
;              A[i] += A[i] * A[i - 1] + A[i + 1];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc.24, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc.24 ], [ 0, %entry ]
  %exitcond5 = icmp ne i64 %indvars.iv, 4
  br i1 %exitcond5, label %for.body, label %for.end.26

for.body:                                         ; preds = %for.cond
  br label %for.cond.1

for.cond.1:                                       ; preds = %for.inc.21, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc22, %for.inc.21 ]
  %exitcond2 = icmp ne i32 %j.0, 4
  br i1 %exitcond2, label %for.body.3, label %for.end.23

for.body.3:                                       ; preds = %for.cond.1
  br label %for.cond.4

for.cond.4:                                       ; preds = %for.inc.18, %for.body.3
  %k.0 = phi i32 [ 0, %for.body.3 ], [ %inc19, %for.inc.18 ]
  %exitcond1 = icmp ne i32 %k.0, 4
  br i1 %exitcond1, label %for.body.6, label %for.end.20

for.body.6:                                       ; preds = %for.cond.4
  br label %for.cond.7

for.cond.7:                                       ; preds = %for.inc, %for.body.6
  %l.0 = phi i32 [ 0, %for.body.6 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %l.0, 4
  br i1 %exitcond, label %for.body.9, label %for.end

for.body.9:                                       ; preds = %for.cond.7
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp = load i32, ptr %arrayidx, align 4
  %tmp6 = add nsw i64 %indvars.iv, -1
  %arrayidx11 = getelementptr inbounds i32, ptr %A, i64 %tmp6
  %tmp7 = load i32, ptr %arrayidx11, align 4
  %mul = mul nsw i32 %tmp, %tmp7
  %tmp8 = add nuw nsw i64 %indvars.iv, 1
  %arrayidx13 = getelementptr inbounds i32, ptr %A, i64 %tmp8
  %tmp9 = load i32, ptr %arrayidx13, align 4
  %add14 = add nsw i32 %mul, %tmp9
  %arrayidx16 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp10 = load i32, ptr %arrayidx16, align 4
  %add17 = add nsw i32 %tmp10, %add14
  store i32 %add17, ptr %arrayidx16, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body.9
  %inc = add nuw nsw i32 %l.0, 1
  br label %for.cond.7

for.end:                                          ; preds = %for.cond.7
  br label %for.inc.18

for.inc.18:                                       ; preds = %for.end
  %inc19 = add nuw nsw i32 %k.0, 1
  br label %for.cond.4

for.end.20:                                       ; preds = %for.cond.4
  br label %for.inc.21

for.inc.21:                                       ; preds = %for.end.20
  %inc22 = add nuw nsw i32 %j.0, 1
  br label %for.cond.1

for.end.23:                                       ; preds = %for.cond.1
  br label %for.inc.24

for.inc.24:                                       ; preds = %for.end.23
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end.26:                                       ; preds = %for.cond
  ret void
}
