; ModuleID = 'tsc_s000.c'
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"
; RUN: llc < %s -march=ppc64 | FileCheck %s

@Y = common global [16000 x double] zeroinitializer, align 32
@X = common global [16000 x double] zeroinitializer, align 32
@Z = common global [16000 x double] zeroinitializer, align 32
@U = common global [16000 x double] zeroinitializer, align 32
@V = common global [16000 x double] zeroinitializer, align 32
@aa = common global [256 x [256 x double]] zeroinitializer, align 32
@bb = common global [256 x [256 x double]] zeroinitializer, align 32
@cc = common global [256 x [256 x double]] zeroinitializer, align 32
@array = common global [65536 x double] zeroinitializer, align 32
@x = common global [16000 x double] zeroinitializer, align 32
@temp = common global double 0.000000e+00, align 8
@temp_int = common global i32 0, align 4
@a = common global [16000 x double] zeroinitializer, align 32
@b = common global [16000 x double] zeroinitializer, align 32
@c = common global [16000 x double] zeroinitializer, align 32
@d = common global [16000 x double] zeroinitializer, align 32
@e = common global [16000 x double] zeroinitializer, align 32
@tt = common global [256 x [256 x double]] zeroinitializer, align 32
@indx = common global [16000 x i32] zeroinitializer, align 32
@xx = common global double* null, align 8
@yy = common global double* null, align 8

define i32 @s000() nounwind {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.end, %entry
  %nl.010 = phi i32 [ 0, %entry ], [ %inc7, %for.end ]
  br label %for.body3

for.body3:                                        ; preds = %for.body3, %for.cond1.preheader
  %indvars.iv = phi i64 [ 0, %for.cond1.preheader ], [ %indvars.iv.next.15, %for.body3 ]
  %arrayidx = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv
  %0 = load double* %arrayidx, align 32, !tbaa !0
  %add = fadd double %0, 1.000000e+00
  %arrayidx5 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv
  store double %add, double* %arrayidx5, align 32, !tbaa !0
  %indvars.iv.next11 = or i64 %indvars.iv, 1
  %arrayidx.1 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next11
  %1 = load double* %arrayidx.1, align 8, !tbaa !0
  %add.1 = fadd double %1, 1.000000e+00
  %arrayidx5.1 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next11
  store double %add.1, double* %arrayidx5.1, align 8, !tbaa !0
  %indvars.iv.next.112 = or i64 %indvars.iv, 2
  %arrayidx.2 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.112
  %2 = load double* %arrayidx.2, align 16, !tbaa !0
  %add.2 = fadd double %2, 1.000000e+00
  %arrayidx5.2 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.112
  store double %add.2, double* %arrayidx5.2, align 16, !tbaa !0
  %indvars.iv.next.213 = or i64 %indvars.iv, 3
  %arrayidx.3 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.213
  %3 = load double* %arrayidx.3, align 8, !tbaa !0
  %add.3 = fadd double %3, 1.000000e+00
  %arrayidx5.3 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.213
  store double %add.3, double* %arrayidx5.3, align 8, !tbaa !0
  %indvars.iv.next.314 = or i64 %indvars.iv, 4
  %arrayidx.4 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.314
  %4 = load double* %arrayidx.4, align 32, !tbaa !0
  %add.4 = fadd double %4, 1.000000e+00
  %arrayidx5.4 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.314
  store double %add.4, double* %arrayidx5.4, align 32, !tbaa !0
  %indvars.iv.next.415 = or i64 %indvars.iv, 5
  %arrayidx.5 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.415
  %5 = load double* %arrayidx.5, align 8, !tbaa !0
  %add.5 = fadd double %5, 1.000000e+00
  %arrayidx5.5 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.415
  store double %add.5, double* %arrayidx5.5, align 8, !tbaa !0
  %indvars.iv.next.516 = or i64 %indvars.iv, 6
  %arrayidx.6 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.516
  %6 = load double* %arrayidx.6, align 16, !tbaa !0
  %add.6 = fadd double %6, 1.000000e+00
  %arrayidx5.6 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.516
  store double %add.6, double* %arrayidx5.6, align 16, !tbaa !0
  %indvars.iv.next.617 = or i64 %indvars.iv, 7
  %arrayidx.7 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.617
  %7 = load double* %arrayidx.7, align 8, !tbaa !0
  %add.7 = fadd double %7, 1.000000e+00
  %arrayidx5.7 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.617
  store double %add.7, double* %arrayidx5.7, align 8, !tbaa !0
  %indvars.iv.next.718 = or i64 %indvars.iv, 8
  %arrayidx.8 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.718
  %8 = load double* %arrayidx.8, align 32, !tbaa !0
  %add.8 = fadd double %8, 1.000000e+00
  %arrayidx5.8 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.718
  store double %add.8, double* %arrayidx5.8, align 32, !tbaa !0
  %indvars.iv.next.819 = or i64 %indvars.iv, 9
  %arrayidx.9 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.819
  %9 = load double* %arrayidx.9, align 8, !tbaa !0
  %add.9 = fadd double %9, 1.000000e+00
  %arrayidx5.9 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.819
  store double %add.9, double* %arrayidx5.9, align 8, !tbaa !0
  %indvars.iv.next.920 = or i64 %indvars.iv, 10
  %arrayidx.10 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.920
  %10 = load double* %arrayidx.10, align 16, !tbaa !0
  %add.10 = fadd double %10, 1.000000e+00
  %arrayidx5.10 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.920
  store double %add.10, double* %arrayidx5.10, align 16, !tbaa !0
  %indvars.iv.next.1021 = or i64 %indvars.iv, 11
  %arrayidx.11 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.1021
  %11 = load double* %arrayidx.11, align 8, !tbaa !0
  %add.11 = fadd double %11, 1.000000e+00
  %arrayidx5.11 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.1021
  store double %add.11, double* %arrayidx5.11, align 8, !tbaa !0
  %indvars.iv.next.1122 = or i64 %indvars.iv, 12
  %arrayidx.12 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.1122
  %12 = load double* %arrayidx.12, align 32, !tbaa !0
  %add.12 = fadd double %12, 1.000000e+00
  %arrayidx5.12 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.1122
  store double %add.12, double* %arrayidx5.12, align 32, !tbaa !0
  %indvars.iv.next.1223 = or i64 %indvars.iv, 13
  %arrayidx.13 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.1223
  %13 = load double* %arrayidx.13, align 8, !tbaa !0
  %add.13 = fadd double %13, 1.000000e+00
  %arrayidx5.13 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.1223
  store double %add.13, double* %arrayidx5.13, align 8, !tbaa !0
  %indvars.iv.next.1324 = or i64 %indvars.iv, 14
  %arrayidx.14 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.1324
  %14 = load double* %arrayidx.14, align 16, !tbaa !0
  %add.14 = fadd double %14, 1.000000e+00
  %arrayidx5.14 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.1324
  store double %add.14, double* %arrayidx5.14, align 16, !tbaa !0
  %indvars.iv.next.1425 = or i64 %indvars.iv, 15
  %arrayidx.15 = getelementptr inbounds [16000 x double]* @Y, i64 0, i64 %indvars.iv.next.1425
  %15 = load double* %arrayidx.15, align 8, !tbaa !0
  %add.15 = fadd double %15, 1.000000e+00
  %arrayidx5.15 = getelementptr inbounds [16000 x double]* @X, i64 0, i64 %indvars.iv.next.1425
  store double %add.15, double* %arrayidx5.15, align 8, !tbaa !0
  %indvars.iv.next.15 = add i64 %indvars.iv, 16
  %lftr.wideiv.15 = trunc i64 %indvars.iv.next.15 to i32
  %exitcond.15 = icmp eq i32 %lftr.wideiv.15, 16000
  br i1 %exitcond.15, label %for.end, label %for.body3

for.end:                                          ; preds = %for.body3
  %call = tail call i32 @dummy(double* getelementptr inbounds ([16000 x double]* @X, i64 0, i64 0), double* getelementptr inbounds ([16000 x double]* @Y, i64 0, i64 0), double* getelementptr inbounds ([16000 x double]* @Z, i64 0, i64 0), double* getelementptr inbounds ([16000 x double]* @U, i64 0, i64 0), double* getelementptr inbounds ([16000 x double]* @V, i64 0, i64 0), [256 x double]* getelementptr inbounds ([256 x [256 x double]]* @aa, i64 0, i64 0), [256 x double]* getelementptr inbounds ([256 x [256 x double]]* @bb, i64 0, i64 0), [256 x double]* getelementptr inbounds ([256 x [256 x double]]* @cc, i64 0, i64 0), double 0.000000e+00) nounwind
  %inc7 = add nsw i32 %nl.010, 1
  %exitcond = icmp eq i32 %inc7, 400000
  br i1 %exitcond, label %for.end8, label %for.cond1.preheader

for.end8:                                         ; preds = %for.end
  ret i32 0

; CHECK: @s000
; CHECK: mtctr
; CHECK: bdnz
}

declare i32 @dummy(double*, double*, double*, double*, double*, [256 x double]*, [256 x double]*, [256 x double]*, double)

!0 = metadata !{metadata !"double", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
