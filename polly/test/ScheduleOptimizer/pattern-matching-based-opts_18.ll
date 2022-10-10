; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-tc-opt=true -debug -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
;      for (i = 0; i < 32; i++)
;        for (j = 0; j < 32; j++)
;          for (k = 0; k < 32; ++k)
;            for (l = 0; l < 32; ++l)
;              for (w = 0; w < 32; ++w)
;                for (q = 0; q < 32; ++q)
;                  C[i][j][k][w] += A[i][l][j][q] * B[q][w][l][k];
;
; CHECK: The tensor contraction pattern was detected
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal void @kernel_tc(i32 %ni, i32 %nj, i32 %nk, i32 %nl, i32 %nq, i32 %nw, double %alpha, double %beta, [32 x [32 x [32 x double]]]* %C, [32 x [32 x [32 x double]]]* %A, [32 x [32 x [32 x double]]]* %B) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc50, %entry
  %indvars.iv71 = phi i64 [ 0, %entry ], [ %indvars.iv.next72, %for.inc50 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc47, %for.cond1.preheader
  %indvars.iv68 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next69, %for.inc47 ]
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.inc44, %for.cond4.preheader
  %indvars.iv65 = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next66, %for.inc44 ]
  br label %for.cond10.preheader

for.cond10.preheader:                             ; preds = %for.inc41, %for.cond7.preheader
  %indvars.iv62 = phi i64 [ 0, %for.cond7.preheader ], [ %indvars.iv.next63, %for.inc41 ]
  br label %for.cond13.preheader

for.cond13.preheader:                             ; preds = %for.inc38, %for.cond10.preheader
  %indvars.iv59 = phi i64 [ 0, %for.cond10.preheader ], [ %indvars.iv.next60, %for.inc38 ]
  br label %for.body15

for.body15:                                       ; preds = %for.body15, %for.cond13.preheader
  %indvars.iv = phi i64 [ 0, %for.cond13.preheader ], [ %indvars.iv.next, %for.body15 ]
  %arrayidx21 = getelementptr inbounds [32 x [32 x [32 x double]]], [32 x [32 x [32 x double]]]* %A, i64 %indvars.iv71, i64 %indvars.iv62, i64 %indvars.iv68, i64 %indvars.iv
  %i = load double, double* %arrayidx21, align 8
  %arrayidx29 = getelementptr inbounds [32 x [32 x [32 x double]]], [32 x [32 x [32 x double]]]* %B, i64 %indvars.iv, i64 %indvars.iv59, i64 %indvars.iv62, i64 %indvars.iv65
  %i1 = load double, double* %arrayidx29, align 8
  %mul = fmul fast double %i1, %i
  %arrayidx37 = getelementptr inbounds [32 x [32 x [32 x double]]], [32 x [32 x [32 x double]]]* %C, i64 %indvars.iv71, i64 %indvars.iv68, i64 %indvars.iv65, i64 %indvars.iv59
  %i2 = load double, double* %arrayidx37, align 8
  %add = fadd fast double %i2, %mul
  store double %add, double* %arrayidx37, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 32
  br i1 %exitcond, label %for.body15, label %for.inc38

for.inc38:                                        ; preds = %for.body15
  %indvars.iv.next60 = add nuw nsw i64 %indvars.iv59, 1
  %exitcond61 = icmp ne i64 %indvars.iv.next60, 32
  br i1 %exitcond61, label %for.cond13.preheader, label %for.inc41

for.inc41:                                        ; preds = %for.inc38
  %indvars.iv.next63 = add nuw nsw i64 %indvars.iv62, 1
  %exitcond64 = icmp ne i64 %indvars.iv.next63, 32
  br i1 %exitcond64, label %for.cond10.preheader, label %for.inc44

for.inc44:                                        ; preds = %for.inc41
  %indvars.iv.next66 = add nuw nsw i64 %indvars.iv65, 1
  %exitcond67 = icmp ne i64 %indvars.iv.next66, 32
  br i1 %exitcond67, label %for.cond7.preheader, label %for.inc47

for.inc47:                                        ; preds = %for.inc44
  %indvars.iv.next69 = add nuw nsw i64 %indvars.iv68, 1
  %exitcond70 = icmp ne i64 %indvars.iv.next69, 32
  br i1 %exitcond70, label %for.cond4.preheader, label %for.inc50

for.inc50:                                        ; preds = %for.inc47
  %indvars.iv.next72 = add nuw nsw i64 %indvars.iv71, 1
  %exitcond73 = icmp ne i64 %indvars.iv.next72, 32
  br i1 %exitcond73, label %for.cond1.preheader, label %for.end52

for.end52:                                        ; preds = %for.inc50
  ret void
}
