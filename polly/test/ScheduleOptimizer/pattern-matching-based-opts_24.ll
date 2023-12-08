; RUN: opt %loadPolly -polly-reschedule=0 -polly-opt-isl \
; RUN: -polly-pattern-matching-based-opts=true -polly-tc-opt=true \
; RUN: -debug -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
;      for (i = 0; i < 1024; i++)
;        for (j = 0; j < 1024; j++)
;          for (l = 0; l < 64; ++l)
;            for (w = 0; w < 64; ++w)
;              C[i][j] += A[i][l][w] * B[w][j][l];
;
; CHECK: The tensor contraction pattern was detected
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal void @kernel_tc(i32 %ni, i32 %nj, i32 %nl, i32 %nq, i32 %nw, double %alpha, double %beta, ptr %C, ptr %A, ptr %B) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc30, %entry
  %indvars.iv43 = phi i64 [ 0, %entry ], [ %indvars.iv.next44, %for.inc30 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc27, %for.cond1.preheader
  %indvars.iv40 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next41, %for.inc27 ]
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.inc24, %for.cond4.preheader
  %indvars.iv37 = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next38, %for.inc24 ]
  br label %for.body9

for.body9:                                        ; preds = %for.body9, %for.cond7.preheader
  %indvars.iv = phi i64 [ 0, %for.cond7.preheader ], [ %indvars.iv.next, %for.body9 ]
  %arrayidx13 = getelementptr inbounds [64 x [64 x double]], ptr %A, i64 %indvars.iv43, i64 %indvars.iv37, i64 %indvars.iv
  %i = load double, ptr %arrayidx13, align 8
  %arrayidx19 = getelementptr inbounds [1024 x [64 x double]], ptr %B, i64 %indvars.iv, i64 %indvars.iv40, i64 %indvars.iv37
  %i1 = load double, ptr %arrayidx19, align 8
  %mul = fmul fast double %i1, %i
  %arrayidx23 = getelementptr inbounds [1024 x double], ptr %C, i64 %indvars.iv43, i64 %indvars.iv40
  %i2 = load double, ptr %arrayidx23, align 8
  %add = fadd fast double %i2, %mul
  store double %add, ptr %arrayidx23, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 64
  br i1 %exitcond, label %for.body9, label %for.inc24

for.inc24:                                        ; preds = %for.body9
  %indvars.iv.next38 = add nuw nsw i64 %indvars.iv37, 1
  %exitcond39 = icmp ne i64 %indvars.iv.next38, 64
  br i1 %exitcond39, label %for.cond7.preheader, label %for.inc27

for.inc27:                                        ; preds = %for.inc24
  %indvars.iv.next41 = add nuw nsw i64 %indvars.iv40, 1
  %exitcond42 = icmp ne i64 %indvars.iv.next41, 1024
  br i1 %exitcond42, label %for.cond4.preheader, label %for.inc30

for.inc30:                                        ; preds = %for.inc27
  %indvars.iv.next44 = add nuw nsw i64 %indvars.iv43, 1
  %exitcond45 = icmp ne i64 %indvars.iv.next44, 1024
  br i1 %exitcond45, label %for.cond1.preheader, label %for.end32

for.end32:                                        ; preds = %for.inc30
  ret void
}
