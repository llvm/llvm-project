; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/flops-4.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/flops-4.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@A0 = dso_local local_unnamed_addr global double 1.000000e+00, align 8
@A1 = dso_local local_unnamed_addr global double 0xBFC5555555559705, align 8
@A2 = dso_local local_unnamed_addr global double 0x3F811111113AE9A3, align 8
@A3 = dso_local local_unnamed_addr global double 0x3F2A01A03FB1CA71, align 8
@A4 = dso_local local_unnamed_addr global double 0x3EC71DF284AA3566, align 8
@A5 = dso_local local_unnamed_addr global double 0x3E5AEB5A8CF8A426, align 8
@A6 = dso_local local_unnamed_addr global double 0x3DE68DF75229C1A6, align 8
@B0 = dso_local local_unnamed_addr global double 1.000000e+00, align 8
@B1 = dso_local local_unnamed_addr global double 0xBFDFFFFFFFFF8156, align 8
@B2 = dso_local local_unnamed_addr global double 0x3FA5555555290224, align 8
@B3 = dso_local local_unnamed_addr global double 0xBF56C16BFFE76516, align 8
@B4 = dso_local local_unnamed_addr global double 0x3EFA019528242DB7, align 8
@B5 = dso_local local_unnamed_addr global double 0xBE927BB3D47DDB8E, align 8
@B6 = dso_local local_unnamed_addr global double 0x3E2157B275DF182A, align 8
@C0 = dso_local local_unnamed_addr global double 1.000000e+00, align 8
@C1 = dso_local local_unnamed_addr global double 0x3FEFFFFFFE37B3E2, align 8
@C2 = dso_local local_unnamed_addr global double 0x3FDFFFFFCC2BA4B8, align 8
@C3 = dso_local local_unnamed_addr global double 0x3FC555587C476915, align 8
@C4 = dso_local local_unnamed_addr global double 0x3FA5555B7E795548, align 8
@C5 = dso_local local_unnamed_addr global double 0x3F810D9A4AD9120C, align 8
@C6 = dso_local local_unnamed_addr global double 0x3F5713187EDB8C05, align 8
@C7 = dso_local local_unnamed_addr global double 0x3F26C077C8173F3A, align 8
@C8 = dso_local local_unnamed_addr global double 0x3F049D03FE04B1CF, align 8
@D1 = dso_local local_unnamed_addr global double 0x3FA47AE143138374, align 8
@D2 = dso_local local_unnamed_addr global double 9.600000e-04, align 8
@D3 = dso_local local_unnamed_addr global double 0x3EB4B05A0FF4A728, align 8
@E2 = dso_local local_unnamed_addr global double 4.800000e-04, align 8
@E3 = dso_local local_unnamed_addr global double 4.110510e-07, align 8
@TLimit = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@piref = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@one = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@two = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@three = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@four = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@five = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@scale = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@sa = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@sb = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@sc = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@.str.4 = private unnamed_addr constant [36 x i8] c"     4   %13.4lf  %10.4lf  %10.4lf\0A\00", align 1
@nulltime = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@TimeArray = dso_local local_unnamed_addr global [3 x double] zeroinitializer, align 8
@T = dso_local local_unnamed_addr global [36 x double] zeroinitializer, align 8
@sd = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@piprg = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@pierr = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@str = private unnamed_addr constant [57 x i8] c"   FLOPS C Program (Double Precision), V2.0 18 Dec 1992\0A\00", align 4
@str.5 = private unnamed_addr constant [47 x i8] c"   Module     Error        RunTime      MFLOPS\00", align 4
@str.6 = private unnamed_addr constant [35 x i8] c"                            (usec)\00", align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = tail call i32 @putchar(i32 10)
  %2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  store double 1.000000e+00, ptr @TLimit, align 8, !tbaa !6
  store double 0x400921FB54442D18, ptr @piref, align 8, !tbaa !6
  store double 1.000000e+00, ptr @one, align 8, !tbaa !6
  store double 2.000000e+00, ptr @two, align 8, !tbaa !6
  store double 3.000000e+00, ptr @three, align 8, !tbaa !6
  store double 4.000000e+00, ptr @four, align 8, !tbaa !6
  store double 5.000000e+00, ptr @five, align 8, !tbaa !6
  store double 1.000000e+00, ptr @scale, align 8, !tbaa !6
  %3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.5)
  %4 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.6)
  %5 = load double, ptr @A3, align 8, !tbaa !6
  %6 = fneg double %5
  store double %6, ptr @A3, align 8, !tbaa !6
  %7 = load double, ptr @A5, align 8, !tbaa !6
  %8 = fneg double %7
  store double %8, ptr @A5, align 8, !tbaa !6
  %9 = load double, ptr @piref, align 8, !tbaa !6
  %10 = load double, ptr @three, align 8, !tbaa !6
  %11 = fmul double %10, 1.562500e+08
  %12 = fdiv double %9, %11
  %13 = load double, ptr @B6, align 8, !tbaa !6
  %14 = load double, ptr @B5, align 8, !tbaa !6
  %15 = load double, ptr @B4, align 8, !tbaa !6
  %16 = load double, ptr @B3, align 8, !tbaa !6
  %17 = load double, ptr @B2, align 8, !tbaa !6
  %18 = load double, ptr @B1, align 8, !tbaa !6
  %19 = load double, ptr @one, align 8, !tbaa !6
  br label %20

20:                                               ; preds = %0, %20
  %21 = phi double [ 0.000000e+00, %0 ], [ %32, %20 ]
  %22 = phi i64 [ 1, %0 ], [ %33, %20 ]
  %23 = uitofp nneg i64 %22 to double
  %24 = fmul double %12, %23
  %25 = fmul double %24, %24
  %26 = tail call double @llvm.fmuladd.f64(double %13, double %25, double %14)
  %27 = tail call double @llvm.fmuladd.f64(double %25, double %26, double %15)
  %28 = tail call double @llvm.fmuladd.f64(double %25, double %27, double %16)
  %29 = tail call double @llvm.fmuladd.f64(double %25, double %28, double %17)
  %30 = tail call double @llvm.fmuladd.f64(double %25, double %29, double %18)
  %31 = tail call double @llvm.fmuladd.f64(double %25, double %30, double %21)
  %32 = fadd double %19, %31
  %33 = add nuw nsw i64 %22, 1
  %34 = icmp eq i64 %33, 156250000
  br i1 %34, label %35, label %20, !llvm.loop !10

35:                                               ; preds = %20
  %36 = fdiv double %9, %10
  %37 = fmul double %36, %36
  %38 = tail call double @llvm.fmuladd.f64(double %13, double %37, double %14)
  %39 = tail call double @llvm.fmuladd.f64(double %37, double %38, double %15)
  %40 = tail call double @llvm.fmuladd.f64(double %37, double %39, double %16)
  %41 = tail call double @llvm.fmuladd.f64(double %37, double %40, double %17)
  %42 = tail call double @llvm.fmuladd.f64(double %37, double %41, double %18)
  %43 = tail call double @llvm.fmuladd.f64(double %37, double %42, double %19)
  %44 = fadd double %19, %43
  %45 = load double, ptr @two, align 8, !tbaa !6
  %46 = tail call double @llvm.fmuladd.f64(double %45, double %32, double %44)
  %47 = fmul double %12, %46
  %48 = fdiv double %47, %45
  store double %48, ptr @sa, align 8, !tbaa !6
  %49 = load double, ptr @A6, align 8, !tbaa !6
  %50 = tail call double @llvm.fmuladd.f64(double %49, double %37, double %8)
  %51 = load double, ptr @A4, align 8, !tbaa !6
  %52 = tail call double @llvm.fmuladd.f64(double %50, double %37, double %51)
  %53 = tail call double @llvm.fmuladd.f64(double %52, double %37, double %6)
  %54 = load double, ptr @A2, align 8, !tbaa !6
  %55 = tail call double @llvm.fmuladd.f64(double %53, double %37, double %54)
  %56 = load double, ptr @A1, align 8, !tbaa !6
  %57 = tail call double @llvm.fmuladd.f64(double %55, double %37, double %56)
  %58 = load double, ptr @A0, align 8, !tbaa !6
  %59 = tail call double @llvm.fmuladd.f64(double %57, double %37, double %58)
  %60 = fmul double %36, %59
  store double %60, ptr @sb, align 8, !tbaa !6
  %61 = fsub double %48, %60
  store double %61, ptr @sc, align 8, !tbaa !6
  %62 = fmul double %61, 1.000000e-30
  %63 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, double noundef %62, double noundef 0.000000e+00, double noundef 0.000000e+00)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
