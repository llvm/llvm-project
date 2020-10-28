; RUN: llc -verify-machineinstrs < %s -enable-misched -mcpu=a2 -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -enable-misched -enable-aa-sched-mi -mcpu=a2 -mtriple=powerpc64-unknown-linux-gnu | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"

@aa = external global [256 x [256 x double]], align 32
@bb = external global [256 x [256 x double]], align 32
@cc = external global [256 x [256 x double]], align 32
@.str1 = external hidden unnamed_addr constant [6 x i8], align 1
@X = external global [16000 x double], align 32
@Y = external global [16000 x double], align 32
@Z = external global [16000 x double], align 32
@U = external global [16000 x double], align 32
@V = external global [16000 x double], align 32
@.str137 = external hidden unnamed_addr constant [14 x i8], align 1

declare void @check(i32 signext) nounwind

declare signext i32 @printf(i8* nocapture, ...) nounwind

declare signext i32 @init(i8*) nounwind

define signext i32 @s000() nounwind {
entry:
  %call = tail call signext i32 @init(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str1, i64 0, i64 0))
  %call1 = tail call i64 @clock() nounwind
  br label %for.cond2.preheader

; CHECK: @s000

for.cond2.preheader:                              ; preds = %for.end, %entry
  %nl.018 = phi i32 [ 0, %entry ], [ %inc9, %for.end ]
  br label %for.body4

for.body4:                                        ; preds = %for.body4, %for.cond2.preheader
  %indvars.iv = phi i64 [ 0, %for.cond2.preheader ], [ %indvars.iv.next.15, %for.body4 ]
  %arrayidx = getelementptr inbounds [16000 x double], [16000 x double]* @Y, i64 0, i64 %indvars.iv
  %arrayidx6 = getelementptr inbounds [16000 x double], [16000 x double]* @X, i64 0, i64 %indvars.iv
  %0 = bitcast double* %arrayidx to <1 x double>*
  %1 = load <1 x double>, <1 x double>* %0, align 32
  %add = fadd <1 x double> %1, <double 1.000000e+00>
  %2 = bitcast double* %arrayidx6 to <1 x double>*
  store <1 x double> %add, <1 x double>* %2, align 32
  %indvars.iv.next.322 = or i64 %indvars.iv, 4
  %arrayidx.4 = getelementptr inbounds [16000 x double], [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.322
  %arrayidx6.4 = getelementptr inbounds [16000 x double], [16000 x double]* @X, i64 0, i64 %indvars.iv.next.322
  %3 = bitcast double* %arrayidx.4 to <1 x double>*
  %4 = load <1 x double>, <1 x double>* %3, align 32
  %add.4 = fadd <1 x double> %4, <double 1.000000e+00>
  %5 = bitcast double* %arrayidx6.4 to <1 x double>*
  store <1 x double> %add.4, <1 x double>* %5, align 32
  %indvars.iv.next.726 = or i64 %indvars.iv, 8
  %arrayidx.8 = getelementptr inbounds [16000 x double], [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.726
  %arrayidx6.8 = getelementptr inbounds [16000 x double], [16000 x double]* @X, i64 0, i64 %indvars.iv.next.726
  %6 = bitcast double* %arrayidx.8 to <1 x double>*
  %7 = load <1 x double>, <1 x double>* %6, align 32
  %add.8 = fadd <1 x double> %7, <double 1.000000e+00>
  %8 = bitcast double* %arrayidx6.8 to <1 x double>*
  store <1 x double> %add.8, <1 x double>* %8, align 32
  %indvars.iv.next.1130 = or i64 %indvars.iv, 12
  %arrayidx.12 = getelementptr inbounds [16000 x double], [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.1130
  %arrayidx6.12 = getelementptr inbounds [16000 x double], [16000 x double]* @X, i64 0, i64 %indvars.iv.next.1130
  %9 = bitcast double* %arrayidx.12 to <1 x double>*
  %10 = load <1 x double>, <1 x double>* %9, align 32
  %add.12 = fadd <1 x double> %10, <double 1.000000e+00>
  %11 = bitcast double* %arrayidx6.12 to <1 x double>*
  store <1 x double> %add.12, <1 x double>* %11, align 32
  %indvars.iv.next.15 = add i64 %indvars.iv, 16
  %lftr.wideiv.15 = trunc i64 %indvars.iv.next.15 to i32
  %exitcond.15 = icmp eq i32 %lftr.wideiv.15, 16000
  br i1 %exitcond.15, label %for.end, label %for.body4

; All of the loads should come before all of the stores.
; CHECK: mtctr
; CHECK: stfd
; CHECK-NOT: lfd
; CHECK: bdnz

for.end:                                          ; preds = %for.body4
  %call7 = tail call signext i32 @dummy(double* getelementptr inbounds ([16000 x double], [16000 x double]* @X, i64 0, i64 0), double* getelementptr inbounds ([16000 x double], [16000 x double]* @Y, i64 0, i64 0), double* getelementptr inbounds ([16000 x double], [16000 x double]* @Z, i64 0, i64 0), double* getelementptr inbounds ([16000 x double], [16000 x double]* @U, i64 0, i64 0), double* getelementptr inbounds ([16000 x double], [16000 x double]* @V, i64 0, i64 0), [256 x double]* getelementptr inbounds ([256 x [256 x double]], [256 x [256 x double]]* @aa, i64 0, i64 0), [256 x double]* getelementptr inbounds ([256 x [256 x double]], [256 x [256 x double]]* @bb, i64 0, i64 0), [256 x double]* getelementptr inbounds ([256 x [256 x double]], [256 x [256 x double]]* @cc, i64 0, i64 0), double 0.000000e+00) nounwind
  %inc9 = add nsw i32 %nl.018, 1
  %exitcond = icmp eq i32 %inc9, 400000
  br i1 %exitcond, label %for.end10, label %for.cond2.preheader

for.end10:                                        ; preds = %for.end
  %call11 = tail call i64 @clock() nounwind
  %sub = sub nsw i64 %call11, %call1
  %conv = sitofp i64 %sub to double
  %div = fdiv double %conv, 1.000000e+06
  %call12 = tail call signext i32 (i8*, ...) @printf(i8* getelementptr inbounds ([14 x i8], [14 x i8]* @.str137, i64 0, i64 0), double %div) nounwind
  tail call void @check(i32 signext 1)
  ret i32 0
}

declare i64 @clock() nounwind

declare signext i32 @dummy(double*, double*, double*, double*, double*, [256 x double]*, [256 x double]*, [256 x double]*, double)
