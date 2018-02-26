; REQUIRES: asserts
; RUN: opt -S -loop-vectorize -debug-only=loop-vectorize -mcpu=skylake %s 2>&1 | FileCheck %s
target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@doublesrc = common local_unnamed_addr global [120 x double] zeroinitializer, align 4
@doubledst = common local_unnamed_addr global [120 x double] zeroinitializer, align 4

; Function Attrs: norecurse nounwind
define void @stride2double(double %k, i32 %width_) {
entry:

; CHECK: Found an estimated cost of 8 for VF 4 For instruction:   %0 = load double
; CHECK: Found an estimated cost of 8 for VF 4 For instruction:   store double

  %cmp27 = icmp sgt i32 %width_, 0
  br i1 %cmp27, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %i.028 = phi i32 [ 0, %for.body.lr.ph ], [ %add16, %for.body ]
  %arrayidx = getelementptr inbounds [120 x double], [120 x double]* @doublesrc, i32 0, i32 %i.028
  %0 = load double, double* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds [120 x double], [120 x double]* @doubledst, i32 0, i32 %i.028
  store double %0, double* %arrayidx2, align 4
  %add4 = add nuw nsw i32 %i.028, 1
  %arrayidx5 = getelementptr inbounds [120 x double], [120 x double]* @doublesrc, i32 0, i32 %add4
  %1 = load double, double* %arrayidx5, align 4
  %arrayidx8 = getelementptr inbounds [120 x double], [120 x double]* @doubledst, i32 0, i32 %add4
  store double %1, double* %arrayidx8, align 4
  %add16 = add nuw nsw i32 %i.028, 2
  %cmp = icmp slt i32 %add16, %width_
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

