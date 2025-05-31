; RUN: opt %loadNPMPolly -passes=polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-tc-opt=true -debug -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
;    for (int i = 0; i < 32; i++)
;      for (int j = 0; j < 32; j++)
;        for (int l = 0; l < 32; l++)
;          for (int w = 0; w < 32; w++)
;            C[i][j] += A[i][l][w] * B[w][j][i];
;
; CHECK-NOT: The tensor contraction pattern was detected
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo(ptr noundef %C, ptr noundef %A, ptr noundef %B) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc33, %entry
  %indvars.iv49 = phi i64 [ 0, %entry ], [ %indvars.iv.next50, %for.inc33 ]
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.inc30, %for.cond1.preheader
  %indvars.iv45 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next46, %for.inc30 ]
  br label %for.cond9.preheader

for.cond9.preheader:                              ; preds = %for.inc27, %for.cond5.preheader
  %indvars.iv41 = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next42, %for.inc27 ]
  br label %for.body12

for.body12:                                       ; preds = %for.body12, %for.cond9.preheader
  %indvars.iv = phi i64 [ 0, %for.cond9.preheader ], [ %indvars.iv.next, %for.body12 ]
  %arrayidx16 = getelementptr inbounds [64 x [64 x double]], ptr %A, i64 %indvars.iv49, i64 %indvars.iv41, i64 %indvars.iv
  %i = load double, ptr %arrayidx16, align 8
  %arrayidx22 = getelementptr inbounds [64 x [64 x double]], ptr %B, i64 %indvars.iv, i64 %indvars.iv45, i64 %indvars.iv49
  %i1 = load double, ptr %arrayidx22, align 8
  %mul = fmul fast double %i1, %i
  %arrayidx26 = getelementptr inbounds [64 x double], ptr %C, i64 %indvars.iv49, i64 %indvars.iv45
  %i2 = load double, ptr %arrayidx26, align 8
  %add = fadd fast double %i2, %mul
  store double %add, ptr %arrayidx26, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 32
  br i1 %exitcond, label %for.body12, label %for.inc27

for.inc27:                                        ; preds = %for.body12
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %exitcond44 = icmp ne i64 %indvars.iv.next42, 32
  br i1 %exitcond44, label %for.cond9.preheader, label %for.inc30

for.inc30:                                        ; preds = %for.inc27
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond48 = icmp ne i64 %indvars.iv.next46, 32
  br i1 %exitcond48, label %for.cond5.preheader, label %for.inc33

for.inc33:                                        ; preds = %for.inc30
  %indvars.iv.next50 = add nuw nsw i64 %indvars.iv49, 1
  %exitcond52 = icmp ne i64 %indvars.iv.next50, 32
  br i1 %exitcond52, label %for.cond1.preheader, label %for.end35

for.end35:                                        ; preds = %for.inc33
  ret void
}
