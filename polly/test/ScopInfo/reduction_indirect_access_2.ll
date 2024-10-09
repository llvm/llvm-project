; RUN: opt %loadPolly -basic-aa -polly-print-scops -disable-output -polly-allow-nonaffine < %s | FileCheck %s
;
; Validate that the accesses to INDICES[i] is not part of a reduction.
;
; CHECK: Reduction Type: NONE
; CHECK: MemRef_INDICES[i0]
; CHECK: Reduction Type: +
; CHECK: MemRef_A[o0]
; CHECK: Reduction Type: +
; CHECK: MemRef_A[o0]
; CHECK: Reduction Type: NONE
; CHECK: MemRef_INDICES[i0]
;
;    void f(double *restrict A, int *restrict INDICES, int N) {
;      for (int i = 0; i < N; i++) {
;        A[INDICES[i]] += N;
;        INDICES[i] += N;
;      }
;    }
;
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"

define void @f(double* noalias %A, i32* noalias %INDICES, i32 %N) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp slt i32 %i.0, %N
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %conv = sitofp i32 %N to double
  %arrayidx = getelementptr inbounds i32, i32* %INDICES, i32 %i.0
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx1 = getelementptr inbounds double, double* %A, i32 %tmp
  %tmp1 = load double, double* %arrayidx1, align 8
  %add = fadd fast double %tmp1, %conv
  store double %add, double* %arrayidx1, align 8
  %add3 = add nsw i32 %tmp, %N
  store i32 %add3, i32* %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add nsw i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
