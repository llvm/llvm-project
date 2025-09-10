; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/flops-8.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/flops-8.c"
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
@.str.4 = private unnamed_addr constant [36 x i8] c"     8   %13.4lf  %10.4lf  %10.4lf\0A\00", align 1
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
  %5 = load double, ptr @piref, align 8, !tbaa !6
  %6 = load double, ptr @three, align 8, !tbaa !6
  %7 = fmul double %6, 1.562500e+08
  %8 = fdiv double %5, %7
  %9 = load double, ptr @B6, align 8, !tbaa !6
  %10 = load double, ptr @B5, align 8, !tbaa !6
  %11 = load double, ptr @B4, align 8, !tbaa !6
  %12 = load double, ptr @B3, align 8, !tbaa !6
  %13 = load double, ptr @B2, align 8, !tbaa !6
  %14 = load double, ptr @B1, align 8, !tbaa !6
  %15 = load double, ptr @one, align 8, !tbaa !6
  %16 = load double, ptr @A6, align 8, !tbaa !6
  %17 = load double, ptr @A5, align 8, !tbaa !6
  %18 = load double, ptr @A4, align 8, !tbaa !6
  %19 = load double, ptr @A3, align 8, !tbaa !6
  %20 = load double, ptr @A2, align 8, !tbaa !6
  %21 = load double, ptr @A1, align 8, !tbaa !6
  %22 = insertelement <2 x double> poison, double %8, i64 0
  %23 = shufflevector <2 x double> %22, <2 x double> poison, <2 x i32> zeroinitializer
  %24 = insertelement <2 x double> poison, double %9, i64 0
  %25 = shufflevector <2 x double> %24, <2 x double> poison, <2 x i32> zeroinitializer
  %26 = insertelement <2 x double> poison, double %10, i64 0
  %27 = shufflevector <2 x double> %26, <2 x double> poison, <2 x i32> zeroinitializer
  %28 = insertelement <2 x double> poison, double %11, i64 0
  %29 = shufflevector <2 x double> %28, <2 x double> poison, <2 x i32> zeroinitializer
  %30 = insertelement <2 x double> poison, double %12, i64 0
  %31 = shufflevector <2 x double> %30, <2 x double> poison, <2 x i32> zeroinitializer
  %32 = insertelement <2 x double> poison, double %13, i64 0
  %33 = shufflevector <2 x double> %32, <2 x double> poison, <2 x i32> zeroinitializer
  %34 = insertelement <2 x double> poison, double %14, i64 0
  %35 = shufflevector <2 x double> %34, <2 x double> poison, <2 x i32> zeroinitializer
  %36 = insertelement <2 x double> poison, double %15, i64 0
  %37 = shufflevector <2 x double> %36, <2 x double> poison, <2 x i32> zeroinitializer
  %38 = insertelement <2 x double> poison, double %16, i64 0
  %39 = shufflevector <2 x double> %38, <2 x double> poison, <2 x i32> zeroinitializer
  %40 = insertelement <2 x double> poison, double %17, i64 0
  %41 = shufflevector <2 x double> %40, <2 x double> poison, <2 x i32> zeroinitializer
  %42 = insertelement <2 x double> poison, double %18, i64 0
  %43 = shufflevector <2 x double> %42, <2 x double> poison, <2 x i32> zeroinitializer
  %44 = insertelement <2 x double> poison, double %19, i64 0
  %45 = shufflevector <2 x double> %44, <2 x double> poison, <2 x i32> zeroinitializer
  %46 = insertelement <2 x double> poison, double %20, i64 0
  %47 = shufflevector <2 x double> %46, <2 x double> poison, <2 x i32> zeroinitializer
  %48 = insertelement <2 x double> poison, double %21, i64 0
  %49 = shufflevector <2 x double> %48, <2 x double> poison, <2 x i32> zeroinitializer
  br label %50

50:                                               ; preds = %50, %0
  %51 = phi i64 [ 0, %0 ], [ %93, %50 ]
  %52 = phi double [ 0.000000e+00, %0 ], [ %92, %50 ]
  %53 = phi <2 x i64> [ <i64 1, i64 2>, %0 ], [ %94, %50 ]
  %54 = add <2 x i64> %53, splat (i64 2)
  %55 = uitofp nneg <2 x i64> %53 to <2 x double>
  %56 = uitofp nneg <2 x i64> %54 to <2 x double>
  %57 = fmul <2 x double> %23, %55
  %58 = fmul <2 x double> %23, %56
  %59 = fmul <2 x double> %57, %57
  %60 = fmul <2 x double> %58, %58
  %61 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %25, <2 x double> %59, <2 x double> %27)
  %62 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %25, <2 x double> %60, <2 x double> %27)
  %63 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %59, <2 x double> %61, <2 x double> %29)
  %64 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %60, <2 x double> %62, <2 x double> %29)
  %65 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %59, <2 x double> %63, <2 x double> %31)
  %66 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %60, <2 x double> %64, <2 x double> %31)
  %67 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %59, <2 x double> %65, <2 x double> %33)
  %68 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %60, <2 x double> %66, <2 x double> %33)
  %69 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %59, <2 x double> %67, <2 x double> %35)
  %70 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %60, <2 x double> %68, <2 x double> %35)
  %71 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %59, <2 x double> %69, <2 x double> %37)
  %72 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %60, <2 x double> %70, <2 x double> %37)
  %73 = fmul <2 x double> %71, %71
  %74 = fmul <2 x double> %72, %72
  %75 = fmul <2 x double> %57, %73
  %76 = fmul <2 x double> %58, %74
  %77 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %39, <2 x double> %59, <2 x double> %41)
  %78 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %39, <2 x double> %60, <2 x double> %41)
  %79 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %77, <2 x double> %59, <2 x double> %43)
  %80 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %78, <2 x double> %60, <2 x double> %43)
  %81 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %79, <2 x double> %59, <2 x double> %45)
  %82 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %80, <2 x double> %60, <2 x double> %45)
  %83 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %81, <2 x double> %59, <2 x double> %47)
  %84 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %82, <2 x double> %60, <2 x double> %47)
  %85 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %83, <2 x double> %59, <2 x double> %49)
  %86 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %84, <2 x double> %60, <2 x double> %49)
  %87 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %85, <2 x double> %59, <2 x double> %37)
  %88 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %86, <2 x double> %60, <2 x double> %37)
  %89 = fmul <2 x double> %75, %87
  %90 = fmul <2 x double> %76, %88
  %91 = tail call double @llvm.vector.reduce.fadd.v2f64(double %52, <2 x double> %89)
  %92 = tail call double @llvm.vector.reduce.fadd.v2f64(double %91, <2 x double> %90)
  %93 = add nuw i64 %51, 4
  %94 = add <2 x i64> %53, splat (i64 4)
  %95 = icmp eq i64 %93, 156249996
  br i1 %95, label %96, label %50, !llvm.loop !10

96:                                               ; preds = %50
  %97 = fmul double %8, 0x41A2A05F1A000000
  %98 = fmul double %97, %97
  %99 = tail call double @llvm.fmuladd.f64(double %9, double %98, double %10)
  %100 = tail call double @llvm.fmuladd.f64(double %98, double %99, double %11)
  %101 = tail call double @llvm.fmuladd.f64(double %98, double %100, double %12)
  %102 = tail call double @llvm.fmuladd.f64(double %98, double %101, double %13)
  %103 = tail call double @llvm.fmuladd.f64(double %98, double %102, double %14)
  %104 = tail call double @llvm.fmuladd.f64(double %98, double %103, double %15)
  %105 = fmul double %104, %104
  %106 = fmul double %97, %105
  %107 = tail call double @llvm.fmuladd.f64(double %16, double %98, double %17)
  %108 = tail call double @llvm.fmuladd.f64(double %107, double %98, double %18)
  %109 = tail call double @llvm.fmuladd.f64(double %108, double %98, double %19)
  %110 = tail call double @llvm.fmuladd.f64(double %109, double %98, double %20)
  %111 = tail call double @llvm.fmuladd.f64(double %110, double %98, double %21)
  %112 = tail call double @llvm.fmuladd.f64(double %111, double %98, double %15)
  %113 = tail call double @llvm.fmuladd.f64(double %106, double %112, double %92)
  %114 = fmul double %8, 0x41A2A05F1C000000
  %115 = fmul double %114, %114
  %116 = tail call double @llvm.fmuladd.f64(double %9, double %115, double %10)
  %117 = tail call double @llvm.fmuladd.f64(double %115, double %116, double %11)
  %118 = tail call double @llvm.fmuladd.f64(double %115, double %117, double %12)
  %119 = tail call double @llvm.fmuladd.f64(double %115, double %118, double %13)
  %120 = tail call double @llvm.fmuladd.f64(double %115, double %119, double %14)
  %121 = tail call double @llvm.fmuladd.f64(double %115, double %120, double %15)
  %122 = fmul double %121, %121
  %123 = fmul double %114, %122
  %124 = tail call double @llvm.fmuladd.f64(double %16, double %115, double %17)
  %125 = tail call double @llvm.fmuladd.f64(double %124, double %115, double %18)
  %126 = tail call double @llvm.fmuladd.f64(double %125, double %115, double %19)
  %127 = tail call double @llvm.fmuladd.f64(double %126, double %115, double %20)
  %128 = tail call double @llvm.fmuladd.f64(double %127, double %115, double %21)
  %129 = tail call double @llvm.fmuladd.f64(double %128, double %115, double %15)
  %130 = tail call double @llvm.fmuladd.f64(double %123, double %129, double %113)
  %131 = fmul double %8, 0x41A2A05F1E000000
  %132 = fmul double %131, %131
  %133 = tail call double @llvm.fmuladd.f64(double %9, double %132, double %10)
  %134 = tail call double @llvm.fmuladd.f64(double %132, double %133, double %11)
  %135 = tail call double @llvm.fmuladd.f64(double %132, double %134, double %12)
  %136 = tail call double @llvm.fmuladd.f64(double %132, double %135, double %13)
  %137 = tail call double @llvm.fmuladd.f64(double %132, double %136, double %14)
  %138 = tail call double @llvm.fmuladd.f64(double %132, double %137, double %15)
  %139 = fmul double %138, %138
  %140 = fmul double %131, %139
  %141 = tail call double @llvm.fmuladd.f64(double %16, double %132, double %17)
  %142 = tail call double @llvm.fmuladd.f64(double %141, double %132, double %18)
  %143 = tail call double @llvm.fmuladd.f64(double %142, double %132, double %19)
  %144 = tail call double @llvm.fmuladd.f64(double %143, double %132, double %20)
  %145 = tail call double @llvm.fmuladd.f64(double %144, double %132, double %21)
  %146 = tail call double @llvm.fmuladd.f64(double %145, double %132, double %15)
  %147 = tail call double @llvm.fmuladd.f64(double %140, double %146, double %130)
  %148 = fdiv double %5, %6
  %149 = fmul double %148, %148
  %150 = tail call double @llvm.fmuladd.f64(double %16, double %149, double %17)
  %151 = tail call double @llvm.fmuladd.f64(double %150, double %149, double %18)
  %152 = tail call double @llvm.fmuladd.f64(double %151, double %149, double %19)
  %153 = tail call double @llvm.fmuladd.f64(double %152, double %149, double %20)
  %154 = tail call double @llvm.fmuladd.f64(double %153, double %149, double %21)
  %155 = tail call double @llvm.fmuladd.f64(double %154, double %149, double %15)
  %156 = fmul double %148, %155
  %157 = tail call double @llvm.fmuladd.f64(double %9, double %149, double %10)
  %158 = tail call double @llvm.fmuladd.f64(double %149, double %157, double %11)
  %159 = tail call double @llvm.fmuladd.f64(double %149, double %158, double %12)
  %160 = tail call double @llvm.fmuladd.f64(double %149, double %159, double %13)
  %161 = tail call double @llvm.fmuladd.f64(double %149, double %160, double %14)
  %162 = tail call double @llvm.fmuladd.f64(double %149, double %161, double %15)
  %163 = fmul double %156, %162
  %164 = fmul double %162, %163
  %165 = load double, ptr @two, align 8, !tbaa !6
  %166 = tail call double @llvm.fmuladd.f64(double %165, double %147, double %164)
  %167 = fmul double %8, %166
  %168 = fdiv double %167, %165
  store double %168, ptr @sa, align 8, !tbaa !6
  store double 0x3FD2AAAAAAAAAAAB, ptr @sb, align 8, !tbaa !6
  %169 = fadd double %168, 0xBFD2AAAAAAAAAAAB
  store double %169, ptr @sc, align 8, !tbaa !6
  %170 = fmul double %169, 1.000000e-30
  %171 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, double noundef %170, double noundef 0.000000e+00, double noundef 0.000000e+00)
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

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.vector.reduce.fadd.v2f64(double, <2 x double>) #4

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree nounwind }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
