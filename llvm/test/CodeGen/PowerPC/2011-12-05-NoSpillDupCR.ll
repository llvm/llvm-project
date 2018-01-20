; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin -mcpu=g4 -disable-ppc-ilp-pref | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=g4 -disable-ppc-ilp-pref | FileCheck %s

; ModuleID = 'tsc.c'
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@a = common global [32000 x float] zeroinitializer, align 16
@b = common global [32000 x float] zeroinitializer, align 16
@c = common global [32000 x float] zeroinitializer, align 16
@d = common global [32000 x float] zeroinitializer, align 16
@e = common global [32000 x float] zeroinitializer, align 16
@aa = common global [256 x [256 x float]] zeroinitializer, align 16
@bb = common global [256 x [256 x float]] zeroinitializer, align 16
@cc = common global [256 x [256 x float]] zeroinitializer, align 16

@.str11 = private unnamed_addr constant [6 x i8] c"s122 \00", align 1
@.str152 = private unnamed_addr constant [14 x i8] c"S122\09 %.2f \09\09\00", align 1

declare i32 @printf(i8* nocapture, ...) nounwind
declare i32 @init(i8* %name) nounwind
declare i64 @clock() nounwind
declare i32 @dummy(float*, float*, float*, float*, float*, [256 x float]*, [256 x float]*, [256 x float]*, float)
declare void @check(i32 %name) nounwind

; CHECK: mfcr
; CHECK: mtcr

define i32 @s122(i32 %n1, i32 %n3) nounwind {
entry:
  %call = tail call i32 @init(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str11, i64 0, i64 0))
  %call1 = tail call i64 @clock() nounwind
  %sub = add nsw i32 %n1, -1
  %cmp316 = icmp slt i32 %sub, 32000
  br i1 %cmp316, label %entry.split.us, label %for.end.7

entry.split.us:                                   ; preds = %entry
  %0 = sext i32 %sub to i64
  %1 = sext i32 %n3 to i64
  br label %for.body4.lr.ph.us

for.body4.us:                                     ; preds = %for.body4.lr.ph.us, %for.body4.us
  %indvars.iv20 = phi i64 [ 0, %for.body4.lr.ph.us ], [ %indvars.iv.next21, %for.body4.us ]
  %indvars.iv = phi i64 [ %0, %for.body4.lr.ph.us ], [ %indvars.iv.next, %for.body4.us ]
  %indvars.iv.next21 = add i64 %indvars.iv20, 1
  %sub5.us = sub i64 31999, %indvars.iv20
  %sext = shl i64 %sub5.us, 32
  %idxprom.us = ashr exact i64 %sext, 32
  %arrayidx.us = getelementptr inbounds [32000 x float], [32000 x float]* @b, i64 0, i64 %idxprom.us
  %2 = load float, float* %arrayidx.us, align 4
  %arrayidx7.us = getelementptr inbounds [32000 x float], [32000 x float]* @a, i64 0, i64 %indvars.iv
  %3 = load float, float* %arrayidx7.us, align 4
  %add8.us = fadd float %3, %2
  store float %add8.us, float* %arrayidx7.us, align 4
  %indvars.iv.next = add i64 %indvars.iv, %1
  %4 = trunc i64 %indvars.iv.next to i32
  %cmp3.us = icmp slt i32 %4, 32000
  br i1 %cmp3.us, label %for.body4.us, label %for.body4.lr.ph.us.1

for.body4.lr.ph.us:                               ; preds = %entry.split.us, %for.end.us.4
  %nl.019.us = phi i32 [ 0, %entry.split.us ], [ %inc.us.4, %for.end.us.4 ]
  br label %for.body4.us

for.end12:                                        ; preds = %for.end.7, %for.end.us.4
  %call13 = tail call i64 @clock() nounwind
  %sub14 = sub nsw i64 %call13, %call1
  %conv = sitofp i64 %sub14 to double
  %div = fdiv double %conv, 1.000000e+06
  %call15 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str152, i64 0, i64 0), double %div) nounwind
  tail call void @check(i32 1)
  ret i32 0

for.body4.lr.ph.us.1:                             ; preds = %for.body4.us
  %call10.us = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  br label %for.body4.us.1

for.body4.us.1:                                   ; preds = %for.body4.us.1, %for.body4.lr.ph.us.1
  %indvars.iv20.1 = phi i64 [ 0, %for.body4.lr.ph.us.1 ], [ %indvars.iv.next21.1, %for.body4.us.1 ]
  %indvars.iv.1 = phi i64 [ %0, %for.body4.lr.ph.us.1 ], [ %indvars.iv.next.1, %for.body4.us.1 ]
  %indvars.iv.next21.1 = add i64 %indvars.iv20.1, 1
  %sub5.us.1 = sub i64 31999, %indvars.iv20.1
  %sext23 = shl i64 %sub5.us.1, 32
  %idxprom.us.1 = ashr exact i64 %sext23, 32
  %arrayidx.us.1 = getelementptr inbounds [32000 x float], [32000 x float]* @b, i64 0, i64 %idxprom.us.1
  %5 = load float, float* %arrayidx.us.1, align 4
  %arrayidx7.us.1 = getelementptr inbounds [32000 x float], [32000 x float]* @a, i64 0, i64 %indvars.iv.1
  %6 = load float, float* %arrayidx7.us.1, align 4
  %add8.us.1 = fadd float %6, %5
  store float %add8.us.1, float* %arrayidx7.us.1, align 4
  %indvars.iv.next.1 = add i64 %indvars.iv.1, %1
  %7 = trunc i64 %indvars.iv.next.1 to i32
  %cmp3.us.1 = icmp slt i32 %7, 32000
  br i1 %cmp3.us.1, label %for.body4.us.1, label %for.body4.lr.ph.us.2

for.body4.lr.ph.us.2:                             ; preds = %for.body4.us.1
  %call10.us.1 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  br label %for.body4.us.2

for.body4.us.2:                                   ; preds = %for.body4.us.2, %for.body4.lr.ph.us.2
  %indvars.iv20.2 = phi i64 [ 0, %for.body4.lr.ph.us.2 ], [ %indvars.iv.next21.2, %for.body4.us.2 ]
  %indvars.iv.2 = phi i64 [ %0, %for.body4.lr.ph.us.2 ], [ %indvars.iv.next.2, %for.body4.us.2 ]
  %indvars.iv.next21.2 = add i64 %indvars.iv20.2, 1
  %sub5.us.2 = sub i64 31999, %indvars.iv20.2
  %sext24 = shl i64 %sub5.us.2, 32
  %idxprom.us.2 = ashr exact i64 %sext24, 32
  %arrayidx.us.2 = getelementptr inbounds [32000 x float], [32000 x float]* @b, i64 0, i64 %idxprom.us.2
  %8 = load float, float* %arrayidx.us.2, align 4
  %arrayidx7.us.2 = getelementptr inbounds [32000 x float], [32000 x float]* @a, i64 0, i64 %indvars.iv.2
  %9 = load float, float* %arrayidx7.us.2, align 4
  %add8.us.2 = fadd float %9, %8
  store float %add8.us.2, float* %arrayidx7.us.2, align 4
  %indvars.iv.next.2 = add i64 %indvars.iv.2, %1
  %10 = trunc i64 %indvars.iv.next.2 to i32
  %cmp3.us.2 = icmp slt i32 %10, 32000
  br i1 %cmp3.us.2, label %for.body4.us.2, label %for.body4.lr.ph.us.3

for.body4.lr.ph.us.3:                             ; preds = %for.body4.us.2
  %call10.us.2 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  br label %for.body4.us.3

for.body4.us.3:                                   ; preds = %for.body4.us.3, %for.body4.lr.ph.us.3
  %indvars.iv20.3 = phi i64 [ 0, %for.body4.lr.ph.us.3 ], [ %indvars.iv.next21.3, %for.body4.us.3 ]
  %indvars.iv.3 = phi i64 [ %0, %for.body4.lr.ph.us.3 ], [ %indvars.iv.next.3, %for.body4.us.3 ]
  %indvars.iv.next21.3 = add i64 %indvars.iv20.3, 1
  %sub5.us.3 = sub i64 31999, %indvars.iv20.3
  %sext25 = shl i64 %sub5.us.3, 32
  %idxprom.us.3 = ashr exact i64 %sext25, 32
  %arrayidx.us.3 = getelementptr inbounds [32000 x float], [32000 x float]* @b, i64 0, i64 %idxprom.us.3
  %11 = load float, float* %arrayidx.us.3, align 4
  %arrayidx7.us.3 = getelementptr inbounds [32000 x float], [32000 x float]* @a, i64 0, i64 %indvars.iv.3
  %12 = load float, float* %arrayidx7.us.3, align 4
  %add8.us.3 = fadd float %12, %11
  store float %add8.us.3, float* %arrayidx7.us.3, align 4
  %indvars.iv.next.3 = add i64 %indvars.iv.3, %1
  %13 = trunc i64 %indvars.iv.next.3 to i32
  %cmp3.us.3 = icmp slt i32 %13, 32000
  br i1 %cmp3.us.3, label %for.body4.us.3, label %for.body4.lr.ph.us.4

for.body4.lr.ph.us.4:                             ; preds = %for.body4.us.3
  %call10.us.3 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  br label %for.body4.us.4

for.body4.us.4:                                   ; preds = %for.body4.us.4, %for.body4.lr.ph.us.4
  %indvars.iv20.4 = phi i64 [ 0, %for.body4.lr.ph.us.4 ], [ %indvars.iv.next21.4, %for.body4.us.4 ]
  %indvars.iv.4 = phi i64 [ %0, %for.body4.lr.ph.us.4 ], [ %indvars.iv.next.4, %for.body4.us.4 ]
  %indvars.iv.next21.4 = add i64 %indvars.iv20.4, 1
  %sub5.us.4 = sub i64 31999, %indvars.iv20.4
  %sext26 = shl i64 %sub5.us.4, 32
  %idxprom.us.4 = ashr exact i64 %sext26, 32
  %arrayidx.us.4 = getelementptr inbounds [32000 x float], [32000 x float]* @b, i64 0, i64 %idxprom.us.4
  %14 = load float, float* %arrayidx.us.4, align 4
  %arrayidx7.us.4 = getelementptr inbounds [32000 x float], [32000 x float]* @a, i64 0, i64 %indvars.iv.4
  %15 = load float, float* %arrayidx7.us.4, align 4
  %add8.us.4 = fadd float %15, %14
  store float %add8.us.4, float* %arrayidx7.us.4, align 4
  %indvars.iv.next.4 = add i64 %indvars.iv.4, %1
  %16 = trunc i64 %indvars.iv.next.4 to i32
  %cmp3.us.4 = icmp slt i32 %16, 32000
  br i1 %cmp3.us.4, label %for.body4.us.4, label %for.end.us.4

for.end.us.4:                                     ; preds = %for.body4.us.4
  %call10.us.4 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  %inc.us.4 = add nsw i32 %nl.019.us, 5
  %exitcond.4 = icmp eq i32 %inc.us.4, 200000
  br i1 %exitcond.4, label %for.end12, label %for.body4.lr.ph.us

for.end.7:                                        ; preds = %entry, %for.end.7
  %nl.019 = phi i32 [ %inc.7, %for.end.7 ], [ 0, %entry ]
  %call10 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  %call10.1 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  %call10.2 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  %call10.3 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  %call10.4 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  %call10.5 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  %call10.6 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  %call10.7 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float 0.000000e+00) nounwind
  %inc.7 = add nsw i32 %nl.019, 8
  %exitcond.7 = icmp eq i32 %inc.7, 200000
  br i1 %exitcond.7, label %for.end12, label %for.end.7
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

declare i32 @puts(i8* nocapture) nounwind

!3 = !{!"branch_weights", i32 64, i32 4}
