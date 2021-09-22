; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; Check that we cast the different pointer types correctly before we compare
; them in the RTC's. We use i8* as max pointer type.
;
; CHECK:   polly.split_new_and_old:
; CHECK:   %polly.access.B = getelementptr float*, float** %B, i64 1024
; CHECK:   %polly.access.A = getelementptr double*, double** %A, i64 0
; CHECK:   %[[paBb:[._a-zA-Z0-9]]] = ptrtoint float** %polly.access.B to i64
; CHECK:   %[[paAb:[._a-zA-Z0-9]]] = ptrtoint double** %polly.access.A to i64
; CHECK:   %[[ALeB:[._a-zA-Z0-9]]] = icmp ule i64 %[[paBb]], %[[paAb]]
; CHECK:   %polly.access.A1 = getelementptr double*, double** %A, i64 1024
; CHECK:   %polly.access.B2 = getelementptr float*, float** %B, i64 0
; CHECK:   %[[paA1b:[._a-zA-Z0-9]]] = ptrtoint double** %polly.access.A1 to i64
; CHECK:   %[[paB2b:[._a-zA-Z0-9]]] = ptrtoint float** %polly.access.B2 to i64
; CHECK:   %[[A1LeB2:[._a-zA-Z0-9]]] = icmp ule i64 %[[paA1b]], %[[paB2b]]
; CHECK:   %[[le1OrLe2:[._a-zA-Z0-9]]] = or i1 %[[ALeB]], %[[A1LeB2]]
; CHECK:   %[[orAndTrue:[._a-zA-Z0-9]]] = and i1 true, %[[le1OrLe2]]
;
;    void jd(double **A, float **B) {
;      for (int i = 0; i < 1024; i++)
;        A[i] = (double *)B[i];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @jd(double** %A, float** %B) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 1024
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float*, float** %B, i64 %indvars.iv
  %tmp = load float*, float** %arrayidx, align 8
  %tmp1 = bitcast float* %tmp to double*
  %arrayidx2 = getelementptr inbounds double*, double** %A, i64 %indvars.iv
  store double* %tmp1, double** %arrayidx2, align 8
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
