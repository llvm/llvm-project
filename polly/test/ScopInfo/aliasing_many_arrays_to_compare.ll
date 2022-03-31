; RUN: opt %loadPolly -polly-print-scops -disable-output \
; RUN:                < %s | FileCheck %s --check-prefix=FOUND
; RUN: opt %loadPolly -polly-print-scops -disable-output \
; RUN:                -polly-rtc-max-arrays-per-group=3 < %s | FileCheck %s \
; RUN:                --check-prefix=IGNORED
;
; FOUND: Function: foo
; IGNORED-NOT: Function: foo
;
;    void foo(float *A, float *B, float *C, float *D) {
;      for (long i = 0; i < 100; i++) {
;        A[i]++;
;        B[i]++;
;        C[i]++;
;        D[i]++;
;      }
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(float* %A, float* %B, float* %C, float* %D) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc7, %for.inc ]
  %exitcond = icmp ne i64 %i.0, 100
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %arrayidx = getelementptr inbounds float, float* %A, i64 %i.0
  %tmp = load float, float* %arrayidx, align 4
  %inc = fadd float %tmp, 1.000000e+00
  store float %inc, float* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds float, float* %B, i64 %i.0
  %tmp1 = load float, float* %arrayidx1, align 4
  %inc2 = fadd float %tmp1, 1.000000e+00
  store float %inc2, float* %arrayidx1, align 4
  %arrayidx3 = getelementptr inbounds float, float* %C, i64 %i.0
  %tmp2 = load float, float* %arrayidx3, align 4
  %inc4 = fadd float %tmp2, 1.000000e+00
  store float %inc4, float* %arrayidx3, align 4
  %arrayidx5 = getelementptr inbounds float, float* %D, i64 %i.0
  %tmp3 = load float, float* %arrayidx5, align 4
  %inc6 = fadd float %tmp3, 1.000000e+00
  store float %inc6, float* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc7 = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
