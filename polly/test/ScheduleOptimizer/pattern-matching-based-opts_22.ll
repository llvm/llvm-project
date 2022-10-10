; RUN: opt %loadPolly -polly-opt-isl -polly-pattern-matching-based-opts=true \
; RUN: -polly-tc-opt=true -debug -disable-output < %s 2>&1 | FileCheck %s
; REQUIRES: asserts
;
;    for (int i = 0; i < 32; i++)
;      for (int j = 0; j < 32; j++)
;        for (int l = 0; l < 32; l++)
;          for (int w = 0; w < 32; w++)
;            C[i][j] += A[i][l][w] * B[w][j][i+3];
;
; CHECK-NOT: The tensor contraction pattern was detected
;
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @foo([64 x double]* noundef %C, [64 x [64 x double]]* noundef %A, [64 x [64 x double]]* noundef %B) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.inc34, %entry
  %indvars.iv50 = phi i64 [ 0, %entry ], [ %indvars.iv.next51, %for.inc34 ]
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.inc31, %for.cond1.preheader
  %indvars.iv46 = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next47, %for.inc31 ]
  br label %for.cond9.preheader

for.cond9.preheader:                              ; preds = %for.inc28, %for.cond5.preheader
  %indvars.iv42 = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next43, %for.inc28 ]
  br label %for.body12

for.body12:                                       ; preds = %for.body12, %for.cond9.preheader
  %indvars.iv = phi i64 [ 0, %for.cond9.preheader ], [ %indvars.iv.next, %for.body12 ]
  %arrayidx16 = getelementptr inbounds [64 x [64 x double]], [64 x [64 x double]]* %A, i64 %indvars.iv50, i64 %indvars.iv42, i64 %indvars.iv
  %i = load double, double* %arrayidx16, align 8
  %i1 = add nuw nsw i64 %indvars.iv50, 3
  %arrayidx22 = getelementptr inbounds [64 x [64 x double]], [64 x [64 x double]]* %B, i64 %indvars.iv, i64 %indvars.iv46, i64 %i1
  %i2 = load double, double* %arrayidx22, align 8
  %mul = fmul fast double %i2, %i
  %arrayidx26 = getelementptr inbounds [64 x double], [64 x double]* %C, i64 %indvars.iv50, i64 %indvars.iv46
  %i3 = load double, double* %arrayidx26, align 8
  %add27 = fadd fast double %i3, %mul
  store double %add27, double* %arrayidx26, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp ne i64 %indvars.iv.next, 32
  br i1 %exitcond, label %for.body12, label %for.inc28

for.inc28:                                        ; preds = %for.body12
  %indvars.iv.next43 = add nuw nsw i64 %indvars.iv42, 1
  %exitcond45 = icmp ne i64 %indvars.iv.next43, 32
  br i1 %exitcond45, label %for.cond9.preheader, label %for.inc31

for.inc31:                                        ; preds = %for.inc28
  %indvars.iv.next47 = add nuw nsw i64 %indvars.iv46, 1
  %exitcond49 = icmp ne i64 %indvars.iv.next47, 32
  br i1 %exitcond49, label %for.cond5.preheader, label %for.inc34

for.inc34:                                        ; preds = %for.inc31
  %indvars.iv.next51 = add nuw nsw i64 %indvars.iv50, 1
  %exitcond54 = icmp ne i64 %indvars.iv.next51, 32
  br i1 %exitcond54, label %for.cond1.preheader, label %for.end36

for.end36:                                        ; preds = %for.inc34
  ret void
}
