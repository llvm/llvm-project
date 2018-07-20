; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin -mcpu=g4 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=g4 | FileCheck %s

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
@temp = common global float 0.000000e+00, align 4

@.str81 = private unnamed_addr constant [6 x i8] c"s3110\00", align 1
@.str235 = private unnamed_addr constant [15 x i8] c"S3110\09 %.2f \09\09\00", align 1

declare i32 @printf(i8* nocapture, ...) nounwind
declare i32 @init(i8* %name) nounwind
declare i64 @clock() nounwind
declare i32 @dummy(float*, float*, float*, float*, float*, [256 x float]*, [256 x float]*, [256 x float]*, float)
declare void @check(i32 %name) nounwind

; CHECK: mfcr
; CHECK: mtcr

define i32 @s3110() nounwind {
entry:
  %call = tail call i32 @init(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str81, i64 0, i64 0))
  %call1 = tail call i64 @clock() nounwind
  br label %for.body

for.body:                                         ; preds = %for.end17, %entry
  %nl.041 = phi i32 [ 0, %entry ], [ %inc22, %for.end17 ]
  %0 = load float, float* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0, i64 0), align 16
  br label %for.cond5.preheader

for.cond5.preheader:                              ; preds = %for.inc15, %for.body
  %indvars.iv42 = phi i64 [ 0, %for.body ], [ %indvars.iv.next43, %for.inc15 ]
  %max.139 = phi float [ %0, %for.body ], [ %max.3.15, %for.inc15 ]
  %xindex.138 = phi i32 [ 0, %for.body ], [ %xindex.3.15, %for.inc15 ]
  %yindex.137 = phi i32 [ 0, %for.body ], [ %yindex.3.15, %for.inc15 ]
  br label %for.body7

for.body7:                                        ; preds = %for.body7, %for.cond5.preheader
  %indvars.iv = phi i64 [ 0, %for.cond5.preheader ], [ %indvars.iv.next.15, %for.body7 ]
  %max.235 = phi float [ %max.139, %for.cond5.preheader ], [ %max.3.15, %for.body7 ]
  %xindex.234 = phi i32 [ %xindex.138, %for.cond5.preheader ], [ %xindex.3.15, %for.body7 ]
  %yindex.233 = phi i32 [ %yindex.137, %for.cond5.preheader ], [ %yindex.3.15, %for.body7 ]
  %arrayidx9 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv
  %1 = load float, float* %arrayidx9, align 16
  %cmp10 = fcmp ogt float %1, %max.235
  %2 = trunc i64 %indvars.iv to i32
  %yindex.3 = select i1 %cmp10, i32 %2, i32 %yindex.233
  %3 = trunc i64 %indvars.iv42 to i32
  %xindex.3 = select i1 %cmp10, i32 %3, i32 %xindex.234
  %max.3 = select i1 %cmp10, float %1, float %max.235
  %indvars.iv.next45 = or i64 %indvars.iv, 1
  %arrayidx9.1 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next45
  %4 = load float, float* %arrayidx9.1, align 4
  %cmp10.1 = fcmp ogt float %4, %max.3
  %5 = trunc i64 %indvars.iv.next45 to i32
  %yindex.3.1 = select i1 %cmp10.1, i32 %5, i32 %yindex.3
  %xindex.3.1 = select i1 %cmp10.1, i32 %3, i32 %xindex.3
  %max.3.1 = select i1 %cmp10.1, float %4, float %max.3
  %indvars.iv.next.146 = or i64 %indvars.iv, 2
  %arrayidx9.2 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.146
  %6 = load float, float* %arrayidx9.2, align 8
  %cmp10.2 = fcmp ogt float %6, %max.3.1
  %7 = trunc i64 %indvars.iv.next.146 to i32
  %yindex.3.2 = select i1 %cmp10.2, i32 %7, i32 %yindex.3.1
  %xindex.3.2 = select i1 %cmp10.2, i32 %3, i32 %xindex.3.1
  %max.3.2 = select i1 %cmp10.2, float %6, float %max.3.1
  %indvars.iv.next.247 = or i64 %indvars.iv, 3
  %arrayidx9.3 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.247
  %8 = load float, float* %arrayidx9.3, align 4
  %cmp10.3 = fcmp ogt float %8, %max.3.2
  %9 = trunc i64 %indvars.iv.next.247 to i32
  %yindex.3.3 = select i1 %cmp10.3, i32 %9, i32 %yindex.3.2
  %xindex.3.3 = select i1 %cmp10.3, i32 %3, i32 %xindex.3.2
  %max.3.3 = select i1 %cmp10.3, float %8, float %max.3.2
  %indvars.iv.next.348 = or i64 %indvars.iv, 4
  %arrayidx9.4 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.348
  %10 = load float, float* %arrayidx9.4, align 16
  %cmp10.4 = fcmp ogt float %10, %max.3.3
  %11 = trunc i64 %indvars.iv.next.348 to i32
  %yindex.3.4 = select i1 %cmp10.4, i32 %11, i32 %yindex.3.3
  %xindex.3.4 = select i1 %cmp10.4, i32 %3, i32 %xindex.3.3
  %max.3.4 = select i1 %cmp10.4, float %10, float %max.3.3
  %indvars.iv.next.449 = or i64 %indvars.iv, 5
  %arrayidx9.5 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.449
  %12 = load float, float* %arrayidx9.5, align 4
  %cmp10.5 = fcmp ogt float %12, %max.3.4
  %13 = trunc i64 %indvars.iv.next.449 to i32
  %yindex.3.5 = select i1 %cmp10.5, i32 %13, i32 %yindex.3.4
  %xindex.3.5 = select i1 %cmp10.5, i32 %3, i32 %xindex.3.4
  %max.3.5 = select i1 %cmp10.5, float %12, float %max.3.4
  %indvars.iv.next.550 = or i64 %indvars.iv, 6
  %arrayidx9.6 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.550
  %14 = load float, float* %arrayidx9.6, align 8
  %cmp10.6 = fcmp ogt float %14, %max.3.5
  %15 = trunc i64 %indvars.iv.next.550 to i32
  %yindex.3.6 = select i1 %cmp10.6, i32 %15, i32 %yindex.3.5
  %xindex.3.6 = select i1 %cmp10.6, i32 %3, i32 %xindex.3.5
  %max.3.6 = select i1 %cmp10.6, float %14, float %max.3.5
  %indvars.iv.next.651 = or i64 %indvars.iv, 7
  %arrayidx9.7 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.651
  %16 = load float, float* %arrayidx9.7, align 4
  %cmp10.7 = fcmp ogt float %16, %max.3.6
  %17 = trunc i64 %indvars.iv.next.651 to i32
  %yindex.3.7 = select i1 %cmp10.7, i32 %17, i32 %yindex.3.6
  %xindex.3.7 = select i1 %cmp10.7, i32 %3, i32 %xindex.3.6
  %max.3.7 = select i1 %cmp10.7, float %16, float %max.3.6
  %indvars.iv.next.752 = or i64 %indvars.iv, 8
  %arrayidx9.8 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.752
  %18 = load float, float* %arrayidx9.8, align 16
  %cmp10.8 = fcmp ogt float %18, %max.3.7
  %19 = trunc i64 %indvars.iv.next.752 to i32
  %yindex.3.8 = select i1 %cmp10.8, i32 %19, i32 %yindex.3.7
  %xindex.3.8 = select i1 %cmp10.8, i32 %3, i32 %xindex.3.7
  %max.3.8 = select i1 %cmp10.8, float %18, float %max.3.7
  %indvars.iv.next.853 = or i64 %indvars.iv, 9
  %arrayidx9.9 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.853
  %20 = load float, float* %arrayidx9.9, align 4
  %cmp10.9 = fcmp ogt float %20, %max.3.8
  %21 = trunc i64 %indvars.iv.next.853 to i32
  %yindex.3.9 = select i1 %cmp10.9, i32 %21, i32 %yindex.3.8
  %xindex.3.9 = select i1 %cmp10.9, i32 %3, i32 %xindex.3.8
  %max.3.9 = select i1 %cmp10.9, float %20, float %max.3.8
  %indvars.iv.next.954 = or i64 %indvars.iv, 10
  %arrayidx9.10 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.954
  %22 = load float, float* %arrayidx9.10, align 8
  %cmp10.10 = fcmp ogt float %22, %max.3.9
  %23 = trunc i64 %indvars.iv.next.954 to i32
  %yindex.3.10 = select i1 %cmp10.10, i32 %23, i32 %yindex.3.9
  %xindex.3.10 = select i1 %cmp10.10, i32 %3, i32 %xindex.3.9
  %max.3.10 = select i1 %cmp10.10, float %22, float %max.3.9
  %indvars.iv.next.1055 = or i64 %indvars.iv, 11
  %arrayidx9.11 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.1055
  %24 = load float, float* %arrayidx9.11, align 4
  %cmp10.11 = fcmp ogt float %24, %max.3.10
  %25 = trunc i64 %indvars.iv.next.1055 to i32
  %yindex.3.11 = select i1 %cmp10.11, i32 %25, i32 %yindex.3.10
  %xindex.3.11 = select i1 %cmp10.11, i32 %3, i32 %xindex.3.10
  %max.3.11 = select i1 %cmp10.11, float %24, float %max.3.10
  %indvars.iv.next.1156 = or i64 %indvars.iv, 12
  %arrayidx9.12 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.1156
  %26 = load float, float* %arrayidx9.12, align 16
  %cmp10.12 = fcmp ogt float %26, %max.3.11
  %27 = trunc i64 %indvars.iv.next.1156 to i32
  %yindex.3.12 = select i1 %cmp10.12, i32 %27, i32 %yindex.3.11
  %xindex.3.12 = select i1 %cmp10.12, i32 %3, i32 %xindex.3.11
  %max.3.12 = select i1 %cmp10.12, float %26, float %max.3.11
  %indvars.iv.next.1257 = or i64 %indvars.iv, 13
  %arrayidx9.13 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.1257
  %28 = load float, float* %arrayidx9.13, align 4
  %cmp10.13 = fcmp ogt float %28, %max.3.12
  %29 = trunc i64 %indvars.iv.next.1257 to i32
  %yindex.3.13 = select i1 %cmp10.13, i32 %29, i32 %yindex.3.12
  %xindex.3.13 = select i1 %cmp10.13, i32 %3, i32 %xindex.3.12
  %max.3.13 = select i1 %cmp10.13, float %28, float %max.3.12
  %indvars.iv.next.1358 = or i64 %indvars.iv, 14
  %arrayidx9.14 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.1358
  %30 = load float, float* %arrayidx9.14, align 8
  %cmp10.14 = fcmp ogt float %30, %max.3.13
  %31 = trunc i64 %indvars.iv.next.1358 to i32
  %yindex.3.14 = select i1 %cmp10.14, i32 %31, i32 %yindex.3.13
  %xindex.3.14 = select i1 %cmp10.14, i32 %3, i32 %xindex.3.13
  %max.3.14 = select i1 %cmp10.14, float %30, float %max.3.13
  %indvars.iv.next.1459 = or i64 %indvars.iv, 15
  %arrayidx9.15 = getelementptr inbounds [256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 %indvars.iv42, i64 %indvars.iv.next.1459
  %32 = load float, float* %arrayidx9.15, align 4
  %cmp10.15 = fcmp ogt float %32, %max.3.14
  %33 = trunc i64 %indvars.iv.next.1459 to i32
  %yindex.3.15 = select i1 %cmp10.15, i32 %33, i32 %yindex.3.14
  %xindex.3.15 = select i1 %cmp10.15, i32 %3, i32 %xindex.3.14
  %max.3.15 = select i1 %cmp10.15, float %32, float %max.3.14
  %indvars.iv.next.15 = add i64 %indvars.iv, 16
  %lftr.wideiv.15 = trunc i64 %indvars.iv.next.15 to i32
  %exitcond.15 = icmp eq i32 %lftr.wideiv.15, 256
  br i1 %exitcond.15, label %for.inc15, label %for.body7

for.inc15:                                        ; preds = %for.body7
  %indvars.iv.next43 = add i64 %indvars.iv42, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next43 to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 256
  br i1 %exitcond, label %for.end17, label %for.cond5.preheader

for.end17:                                        ; preds = %for.inc15
  %conv = sitofp i32 %xindex.3.15 to float
  %add = fadd float %max.3.15, %conv
  %conv18 = sitofp i32 %yindex.3.15 to float
  %add19 = fadd float %add, %conv18
  %call20 = tail call i32 @dummy(float* getelementptr inbounds ([32000 x float], [32000 x float]* @a, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @b, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @c, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @d, i64 0, i64 0), float* getelementptr inbounds ([32000 x float], [32000 x float]* @e, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @aa, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @bb, i64 0, i64 0), [256 x float]* getelementptr inbounds ([256 x [256 x float]], [256 x [256 x float]]* @cc, i64 0, i64 0), float %add19) nounwind
  %inc22 = add nsw i32 %nl.041, 1
  %exitcond44 = icmp eq i32 %inc22, 78100
  br i1 %exitcond44, label %for.end23, label %for.body

for.end23:                                        ; preds = %for.end17
  %call24 = tail call i64 @clock() nounwind
  %sub = sub nsw i64 %call24, %call1
  %conv25 = sitofp i64 %sub to double
  %div = fdiv double %conv25, 1.000000e+06
  %call26 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str235, i64 0, i64 0), double %div) nounwind
  %add29 = fadd float %add, 1.000000e+00
  %add31 = fadd float %add29, %conv18
  %add32 = fadd float %add31, 1.000000e+00
  store float %add32, float* @temp, align 4
  tail call void @check(i32 -1)
  ret i32 0
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1) nounwind

declare i32 @puts(i8* nocapture) nounwind

!3 = !{!"branch_weights", i32 64, i32 4}
