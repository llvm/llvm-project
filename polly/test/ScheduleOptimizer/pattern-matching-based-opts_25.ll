; RUN: opt %loadNPMPolly -passes=polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-tc-opt=true -debug -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
;    for (int i = 0; i < 32; i++)
;      for (int j = 0; j < 32; j++)
;        for (int w = 0; w < 32; w++)
;          C[i][j] += A[i][w] * B[w][j][i];
;
; CHECK-NOT: The tensor contraction pattern was detected
;
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx12.0.0"

define void @foo(ptr noundef %C, ptr noundef %A, ptr noundef %B) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond.cleanup3, %entry
  %indvars.iv45 = phi i64 [ 0, %entry ], [ %indvars.iv.next46, %for.cond.cleanup3 ]
  br label %for.cond5.preheader

for.cond.cleanup:                                 ; preds = %for.cond.cleanup3
  ret void

for.cond5.preheader:                              ; preds = %for.cond.cleanup7, %for.cond1.preheader
  %indvars.iv41 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next42, %for.cond.cleanup7 ]
  %arrayidx20 = getelementptr inbounds [64 x double], ptr %C, i64 %indvars.iv45, i64 %indvars.iv41
  %.pre = load double, ptr %arrayidx20, align 8
  br label %for.body8

for.cond.cleanup3:                                ; preds = %for.cond.cleanup7
  %indvars.iv.next46 = add nuw nsw i64 %indvars.iv45, 1
  %exitcond48.not = icmp eq i64 %indvars.iv.next46, 32
  br i1 %exitcond48.not, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup7:                                ; preds = %for.body8
  %indvars.iv.next42 = add nuw nsw i64 %indvars.iv41, 1
  %exitcond44.not = icmp eq i64 %indvars.iv.next42, 32
  br i1 %exitcond44.not, label %for.cond.cleanup3, label %for.cond5.preheader

for.body8:                                        ; preds = %for.body8, %for.cond5.preheader
  %i = phi double [ %.pre, %for.cond5.preheader ], [ %i3, %for.body8 ]
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next, %for.body8 ]
  %arrayidx10 = getelementptr inbounds [64 x double], ptr %A, i64 %indvars.iv45, i64 %indvars.iv
  %i1 = load double, ptr %arrayidx10, align 8
  %arrayidx16 = getelementptr inbounds [64 x [64 x double]], ptr %B, i64 %indvars.iv, i64 %indvars.iv41, i64 %indvars.iv45
  %i2 = load double, ptr %arrayidx16, align 8
  %i3 = tail call double @llvm.fmuladd.f64(double %i1, double %i2, double %i)
  store double %i3, ptr %arrayidx20, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 32
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8
}

declare double @llvm.fmuladd.f64(double, double, double)
