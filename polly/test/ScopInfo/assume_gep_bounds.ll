; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -polly-print-function-scops -disable-output < %s | FileCheck %s

;    void foo(float A[][20][30], long n, long m, long p) {
;      for (long i = 0; i < n; i++)
;        for (long j = 0; j < m; j++)
;          for (long k = 0; k < p; k++)
;            A[i][j][k] = i + j + k;
;    }

; For the above code we want to assume that all memory accesses are within the
; bounds of the array A. In C (and LLVM-IR) this is not required, such that out
; of bounds accesses are valid. However, as such accesses are uncommon, cause
; complicated dependence pattern and as a result make dependence analysis more
; costly and may prevent or hinder useful program transformations, we assume
; absence of out-of-bound accesses. To do so we derive the set of parameter
; values for which our assumption holds.

; CHECK: Assumed Context
; CHECK-NEXT: [n, m, p] -> {  :
; CHECK-DAG:                    p <= 30
; CHECK-DAG:                     and
; CHECK-DAG:                    m <= 20
; CHECK:                   }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo([20 x [30 x float]]* %A, i64 %n, i64 %m, i64 %p) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc13, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc14, %for.inc13 ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end15

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc10, %for.body
  %j.0 = phi i64 [ 0, %for.body ], [ %inc11, %for.inc10 ]
  %cmp2 = icmp slt i64 %j.0, %m
  br i1 %cmp2, label %for.body3, label %for.end12

for.body3:                                        ; preds = %for.cond1
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i64 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %cmp5 = icmp slt i64 %k.0, %p
  br i1 %cmp5, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %add = add nsw i64 %i.0, %j.0
  %add7 = add nsw i64 %add, %k.0
  %conv = sitofp i64 %add7 to float
  %arrayidx9 = getelementptr inbounds [20 x [30 x float]], [20 x [30 x float]]* %A, i64 %i.0, i64 %j.0, i64 %k.0
  store float %conv, float* %arrayidx9, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %inc = add nsw i64 %k.0, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  br label %for.inc10

for.inc10:                                        ; preds = %for.end
  %inc11 = add nsw i64 %j.0, 1
  br label %for.cond1

for.end12:                                        ; preds = %for.cond1
  br label %for.inc13

for.inc13:                                        ; preds = %for.end12
  %inc14 = add nsw i64 %i.0, 1
  br label %for.cond

for.end15:                                        ; preds = %for.cond
  ret void
}
