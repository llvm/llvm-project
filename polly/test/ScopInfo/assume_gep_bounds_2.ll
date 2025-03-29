; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 \
; RUN:  -polly-precise-inbounds | FileCheck %s
;
;    void foo(float A[restrict][20], float B[restrict][20], long n, long m,
;             long p) {
;      for (long i = 0; i < n; i++)
;        for (long j = 0; j < m; j++)
;          A[i][j] = i + j;
;      for (long i = 0; i < m; i++)
;        for (long j = 0; j < p; j++)
;          B[i][j] = i + j;
;    }

; This code is within bounds either if m and p are smaller than the array sizes,
; but also if only p is smaller than the size of the second B dimension and n
; is such that the first loop is never executed and consequently A is never
; accessed. In this case the value of m does not matter.

; CHECK:      Assumed Context:
; CHECK-NEXT: [n, m, p] -> {  : p <= 20 and (n <= 0 or (n > 0 and m <= 20)) }

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @foo(ptr noalias %A, ptr noalias %B, i64 %n, i64 %m, i64 %p) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc5, %entry
  %i.0 = phi i64 [ 0, %entry ], [ %inc6, %for.inc5 ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end7

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %j.0 = phi i64 [ 0, %for.body ], [ %inc, %for.inc ]
  %cmp2 = icmp slt i64 %j.0, %m
  br i1 %cmp2, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %add = add nsw i64 %i.0, %j.0
  %conv = sitofp i64 %add to float
  %arrayidx4 = getelementptr inbounds [20 x float], ptr %A, i64 %i.0, i64 %j.0
  store float %conv, ptr %arrayidx4, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %inc = add nsw i64 %j.0, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc5

for.inc5:                                         ; preds = %for.end
  %inc6 = add nsw i64 %i.0, 1
  br label %for.cond

for.end7:                                         ; preds = %for.cond
  br label %for.cond9

for.cond9:                                        ; preds = %for.inc25, %for.end7
  %i8.0 = phi i64 [ 0, %for.end7 ], [ %inc26, %for.inc25 ]
  %cmp10 = icmp slt i64 %i8.0, %m
  br i1 %cmp10, label %for.body12, label %for.end27

for.body12:                                       ; preds = %for.cond9
  br label %for.cond14

for.cond14:                                       ; preds = %for.inc22, %for.body12
  %j13.0 = phi i64 [ 0, %for.body12 ], [ %inc23, %for.inc22 ]
  %cmp15 = icmp slt i64 %j13.0, %p
  br i1 %cmp15, label %for.body17, label %for.end24

for.body17:                                       ; preds = %for.cond14
  %add18 = add nsw i64 %i8.0, %j13.0
  %conv19 = sitofp i64 %add18 to float
  %arrayidx21 = getelementptr inbounds [20 x float], ptr %B, i64 %i8.0, i64 %j13.0
  store float %conv19, ptr %arrayidx21, align 4
  br label %for.inc22

for.inc22:                                        ; preds = %for.body17
  %inc23 = add nsw i64 %j13.0, 1
  br label %for.cond14

for.end24:                                        ; preds = %for.cond14
  br label %for.inc25

for.inc25:                                        ; preds = %for.end24
  %inc26 = add nsw i64 %i8.0, 1
  br label %for.cond9

for.end27:                                        ; preds = %for.cond9
  ret void
}
