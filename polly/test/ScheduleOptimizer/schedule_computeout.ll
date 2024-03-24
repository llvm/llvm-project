; RUN: opt %loadPolly -S -polly-optree -polly-delicm  -polly-opt-isl -polly-schedule-computeout=10000 -debug-only="polly-opt-isl" < %s 2>&1 | FileCheck %s
; REQUIRES: asserts

; Bailout if the computations of schedule compute exceeds the max scheduling quota.
; Max compute out is initialized to 300000, Here it is set to 10000 for test purpose.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global ptr null, align 8
@b = dso_local local_unnamed_addr global ptr null, align 8
@c = dso_local local_unnamed_addr global ptr null, align 8

define dso_local void @foo(i32 noundef %I, i32 noundef %J, i32 noundef %K1, i32 noundef %K2, i32 noundef %L1, i32 noundef %L2) local_unnamed_addr {
entry:
  %j = alloca i32, align 4
  store volatile i32 0, ptr %j, align 4
  %j.0.j.0.j.0.54 = load volatile i32, ptr %j, align 4
  %cmp55 = icmp slt i32 %j.0.j.0.j.0.54, %J
  br i1 %cmp55, label %for.body.lr.ph, label %for.cond.cleanup

for.body.lr.ph:                                   ; preds = %entry
  %0 = load ptr, ptr @a, align 8
  %1 = load ptr, ptr @b, align 8
  %2 = load ptr, ptr %1, align 8
  %cmp352 = icmp slt i32 %L1, %L2
  %cmp750 = icmp slt i32 %K1, %K2
  %3 = sext i32 %K1 to i64
  %4 = sext i32 %L1 to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.cond.cleanup4, %entry
  ret void

for.body:                                         ; preds = %for.cond.cleanup4, %for.body.lr.ph
  br i1 %cmp352, label %for.cond6.preheader.preheader, label %for.cond.cleanup4

for.cond6.preheader.preheader:                    ; preds = %for.body
  %wide.trip.count66 = sext i32 %L2 to i64
  br label %for.cond6.preheader

for.cond6.preheader:                              ; preds = %for.cond.cleanup8, %for.cond6.preheader.preheader
  %indvars.iv61 = phi i64 [ %4, %for.cond6.preheader.preheader ], [ %indvars.iv.next62, %for.cond.cleanup8 ]
  br i1 %cmp750, label %for.cond10.preheader.lr.ph, label %for.cond.cleanup8

for.cond10.preheader.lr.ph:                       ; preds = %for.cond6.preheader
  %5 = mul nsw i64 %indvars.iv61, 516
  %6 = mul nsw i64 %indvars.iv61, 516
  %wide.trip.count = sext i32 %K2 to i64
  br label %for.cond10.preheader

for.cond.cleanup4:                                ; preds = %for.cond.cleanup8, %for.body
  %j.0.j.0.j.0.45 = load volatile i32, ptr %j, align 4
  %inc34 = add nsw i32 %j.0.j.0.j.0.45, 1
  store volatile i32 %inc34, ptr %j, align 4
  %j.0.j.0.j.0. = load volatile i32, ptr %j, align 4
  %cmp = icmp slt i32 %j.0.j.0.j.0., %J
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond10.preheader:                             ; preds = %for.cond.cleanup12, %for.cond10.preheader.lr.ph
  %indvars.iv = phi i64 [ %3, %for.cond10.preheader.lr.ph ], [ %indvars.iv.next, %for.cond.cleanup12 ]
  %7 = getelementptr float, ptr %0, i64 %indvars.iv
  %arrayidx18 = getelementptr float, ptr %7, i64 %5
  %8 = load float, ptr %arrayidx18, align 4
  br label %for.cond14.preheader

for.cond.cleanup8:                                ; preds = %for.cond.cleanup12, %for.cond6.preheader
  %indvars.iv.next62 = add nsw i64 %indvars.iv61, 1
  %exitcond67.not = icmp eq i64 %indvars.iv.next62, %wide.trip.count66
  br i1 %exitcond67.not, label %for.cond.cleanup4, label %for.cond6.preheader

for.cond14.preheader:                             ; preds = %for.cond.cleanup16, %for.cond10.preheader
  %m.049 = phi i32 [ -2, %for.cond10.preheader ], [ %inc21, %for.cond.cleanup16 ]
  %sum.048 = phi float [ 0.000000e+00, %for.cond10.preheader ], [ %add19, %for.cond.cleanup16 ]
  br label %for.body17

for.cond.cleanup12:                               ; preds = %for.cond.cleanup16
  %9 = getelementptr float, ptr %2, i64 %indvars.iv
  %arrayidx26 = getelementptr float, ptr %9, i64 %6
  store float %add19, ptr %arrayidx26, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %exitcond60.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond60.not, label %for.cond.cleanup8, label %for.cond10.preheader

for.cond.cleanup16:                               ; preds = %for.body17
  %inc21 = add nsw i32 %m.049, 1
  %exitcond56.not = icmp eq i32 %inc21, 3
  br i1 %exitcond56.not, label %for.cond.cleanup12, label %for.cond14.preheader

for.body17:                                       ; preds = %for.body17, %for.cond14.preheader
  %n.047 = phi i32 [ -2, %for.cond14.preheader ], [ %inc, %for.body17 ]
  %sum.146 = phi float [ %sum.048, %for.cond14.preheader ], [ %add19, %for.body17 ]
  %add19 = fadd float %sum.146, %8
  %inc = add nsw i32 %n.047, 1
  %exitcond.not = icmp eq i32 %inc, 3
  br i1 %exitcond.not, label %for.cond.cleanup16, label %for.body17
}

; CHECK: Schedule optimizer calculation exceeds ISL quota
