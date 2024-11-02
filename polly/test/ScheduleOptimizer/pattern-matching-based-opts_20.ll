; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-tc-opt=true -debug -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
;   for (i = 0; i < 16; i++)
;     for (j = 0; j < 16; j++)
;       for (k = 0; k < 8; ++k)
;         for (l = 0; l < 1024; ++l)
;           for (w = 0; w < 8; ++w)
;             for (q = 0; q < 8; ++q)
;               for (x = 0; x < 8; ++x)
;                 C[i][j][k][w][q][x] += A[l][x][j][k] * B[w][q][l][i];
;
; CHECK: The tensor contraction pattern was detected
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal void @kernel_tc([16 x [8 x [8 x [8 x [8 x double]]]]]* %C, [8 x [16 x [8 x double]]]* %A, [8 x [1024 x [16 x double]]]* %B) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc60, %entry
  %indvars.iv85 = phi i64 [ 0, %entry ], [ %indvars.iv.next86, %for.inc60 ]
  br label %for.cond4.preheader

for.cond4.preheader:                              ; preds = %for.inc57, %for.cond1.preheader
  %indvars.iv82 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next83, %for.inc57 ]
  br label %for.cond7.preheader

for.cond7.preheader:                              ; preds = %for.inc54, %for.cond4.preheader
  %indvars.iv79 = phi i64 [ 0, %for.cond4.preheader ], [ %indvars.iv.next80, %for.inc54 ]
  br label %for.cond10.preheader

for.cond10.preheader:                             ; preds = %for.inc51, %for.cond7.preheader
  %indvars.iv76 = phi i64 [ 0, %for.cond7.preheader ], [ %indvars.iv.next77, %for.inc51 ]
  br label %for.cond13.preheader

for.cond13.preheader:                             ; preds = %for.inc48, %for.cond10.preheader
  %indvars.iv73 = phi i64 [ 0, %for.cond10.preheader ], [ %indvars.iv.next74, %for.inc48 ]
  br label %for.cond16.preheader

for.cond16.preheader:                             ; preds = %for.inc45, %for.cond13.preheader
  %indvars.iv70 = phi i64 [ 0, %for.cond13.preheader ], [ %indvars.iv.next71, %for.inc45 ]
  br label %for.body18

for.body18:                                       ; preds = %for.body18, %for.cond16.preheader
  %indvars.iv = phi i64 [ 0, %for.cond16.preheader ], [ %indvars.iv.next, %for.body18 ]
  %arrayidx24 = getelementptr inbounds [8 x [16 x [8 x double]]], [8 x [16 x [8 x double]]]* %A, i64 %indvars.iv76, i64 %indvars.iv, i64 %indvars.iv82, i64 %indvars.iv79
  %i = load double, double* %arrayidx24, align 8
  %arrayidx32 = getelementptr inbounds [8 x [1024 x [16 x double]]], [8 x [1024 x [16 x double]]]* %B, i64 %indvars.iv73, i64 %indvars.iv70, i64 %indvars.iv76, i64 %indvars.iv85
  %i1 = load double, double* %arrayidx32, align 8
  %mul = fmul fast double %i1, %i
  %arrayidx44 = getelementptr inbounds [16 x [8 x [8 x [8 x [8 x double]]]]], [16 x [8 x [8 x [8 x [8 x double]]]]]* %C, i64 %indvars.iv85, i64 %indvars.iv82, i64 %indvars.iv79, i64 %indvars.iv73, i64 %indvars.iv70, i64 %indvars.iv
  %i2 = load double, double* %arrayidx44, align 8
  %add = fadd fast double %i2, %mul
  store double %add, double* %arrayidx44, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 8
  br i1 %exitcond, label %for.body18, label %for.inc45

for.inc45:                                        ; preds = %for.body18
  %indvars.iv.next71 = add nuw nsw i64 %indvars.iv70, 1
  %exitcond72 = icmp ne i64 %indvars.iv.next71, 8
  br i1 %exitcond72, label %for.cond16.preheader, label %for.inc48

for.inc48:                                        ; preds = %for.inc45
  %indvars.iv.next74 = add nuw nsw i64 %indvars.iv73, 1
  %exitcond75 = icmp ne i64 %indvars.iv.next74, 8
  br i1 %exitcond75, label %for.cond13.preheader, label %for.inc51

for.inc51:                                        ; preds = %for.inc48
  %indvars.iv.next77 = add nuw nsw i64 %indvars.iv76, 1
  %exitcond78 = icmp ne i64 %indvars.iv.next77, 1024
  br i1 %exitcond78, label %for.cond10.preheader, label %for.inc54

for.inc54:                                        ; preds = %for.inc51
  %indvars.iv.next80 = add nuw nsw i64 %indvars.iv79, 1
  %exitcond81 = icmp ne i64 %indvars.iv.next80, 8
  br i1 %exitcond81, label %for.cond7.preheader, label %for.inc57

for.inc57:                                        ; preds = %for.inc54
  %indvars.iv.next83 = add nuw nsw i64 %indvars.iv82, 1
  %exitcond84 = icmp ne i64 %indvars.iv.next83, 16
  br i1 %exitcond84, label %for.cond4.preheader, label %for.inc60

for.inc60:                                        ; preds = %for.inc57
  %indvars.iv.next86 = add nuw nsw i64 %indvars.iv85, 1
  %exitcond87 = icmp ne i64 %indvars.iv.next86, 16
  br i1 %exitcond87, label %for.cond1.preheader, label %for.end62

for.end62:                                        ; preds = %for.inc60
  ret void
}
