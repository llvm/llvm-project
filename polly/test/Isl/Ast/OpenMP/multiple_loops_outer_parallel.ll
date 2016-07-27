; RUN: opt %loadPolly -polly-ast -polly-parallel -polly-parallel-force -analyze < %s | FileCheck %s
; RUN: opt %loadPolly -polyhedral-info -polly-check-parallel -analyze < %s | FileCheck %s -check-prefix=PINFO
;
;       void jd(int *A) {
; CHECK:  #pragma omp parallel for
; PINFO:  for.cond2: Loop is parallel.
;         for (int i = 0; i < 1024; i++)
;           A[i] = 1;
; CHECK:  #pragma omp parallel for
; PINFO:  for.cond: Loop is parallel.
;         for (int i = 0; i < 1024; i++)
;           A[i] = A[i] * 2;
;       }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(i32* %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc ], [ 0, %entry ]
  %exitcond3 = icmp ne i64 %indvars.iv1, 1024
  br i1 %exitcond3, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv1
  store i32 1, i32* %arrayidx, align 4
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
  %arrayidx6 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp = load i32, i32* %arrayidx6, align 4
  %mul = shl nsw i32 %tmp, 1
  %arrayidx8 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %mul, i32* %arrayidx8, align 4
  br label %for.inc9

for.inc9:                                         ; preds = %for.body4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond2

for.end11:                                        ; preds = %for.cond2
  ret void
}
