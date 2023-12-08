; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s
;
; Region with an exit node that has a PHI node multiple incoming edges from
; inside the region. Motivation for supporting such cases in Polly.
;
;    float test(long n, float A[static const restrict n]) {
;      float sum = 0;
;      for (long i = 0; i < n; i += 1)
;        sum += A[i];
;      for (long i = 0; i < n; i += 1)
;        sum += A[i];
;      return sum;
;    }
;
;

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define float @test(i64 %n, ptr noalias nonnull %A) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %sum.0 = phi float [ 0.000000e+00, %entry ], [ %add, %for.inc ]
  %i.0 = phi i64 [ 0, %entry ], [ %add1, %for.inc ]
  %cmp = icmp slt i64 %i.0, %n
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds float, ptr %A, i64 %i.0
  %tmp = load float, ptr %arrayidx, align 4
  %add = fadd float %sum.0, %tmp
  %add1 = add nuw nsw i64 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %sum.0.lcssa = phi float [ %sum.0, %for.cond ]
  br label %for.cond.3

for.cond.3:                                       ; preds = %for.inc.8, %for.end
  %sum.1 = phi float [ %sum.0.lcssa, %for.end ], [ %add7, %for.inc.8 ]
  %i2.0 = phi i64 [ 0, %for.end ], [ %add9, %for.inc.8 ]
  %cmp4 = icmp slt i64 %i2.0, %n
  br i1 %cmp4, label %for.body.5, label %for.end.10

for.body.5:                                       ; preds = %for.cond.3
  br label %for.inc.8

for.inc.8:                                        ; preds = %for.body.5
  %arrayidx6 = getelementptr inbounds float, ptr %A, i64 %i2.0
  %tmp1 = load float, ptr %arrayidx6, align 4
  %add7 = fadd float %sum.1, %tmp1
  %add9 = add nuw nsw i64 %i2.0, 1
  br label %for.cond.3

for.end.10:                                       ; preds = %for.cond.3
  %sum.1.lcssa = phi float [ %sum.1, %for.cond.3 ]
  ret float %sum.1.lcssa
}

; CHECK: Valid Region for Scop: for.cond => for.end.10
