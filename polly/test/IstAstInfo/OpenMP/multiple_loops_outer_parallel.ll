; RUN: opt %loadPolly -polly-print-ast -polly-parallel -polly-parallel-force -disable-output < %s | FileCheck %s
;
;       void jd(int *A) {
; CHECK:  #pragma omp parallel for
;         for (int i = 0; i < 1024; i++)
;           A[i] = 1;
; CHECK:  #pragma omp parallel for
;         for (int i = 0; i < 1024; i++)
;           A[i] = A[i] * 2;
;       }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(ptr %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc ], [ 0, %entry ]
  %exitcond3 = icmp ne i64 %indvars.iv1, 1024
  br i1 %exitcond3, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv1
  store i32 1, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc9, %for.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc9 ], [ 0, %for.end ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body4, label %for.end11

for.body4:                                        ; preds = %for.cond2
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp = load i32, ptr %arrayidx6, align 4
  %mul = shl nsw i32 %tmp, 1
  %arrayidx8 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %mul, ptr %arrayidx8, align 4
  br label %for.inc9

for.inc9:                                         ; preds = %for.body4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond2

for.end11:                                        ; preds = %for.cond2
  ret void
}
