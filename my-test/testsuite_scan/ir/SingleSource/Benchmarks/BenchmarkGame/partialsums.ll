; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/partialsums.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/partialsums.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [9 x i8] c"%.9f\09%s\0A\00", align 1
@.str.1 = private unnamed_addr constant [8 x i8] c"(2/3)^k\00", align 1
@.str.2 = private unnamed_addr constant [7 x i8] c"k^-0.5\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"1/k(k+1)\00", align 1
@.str.4 = private unnamed_addr constant [12 x i8] c"Flint Hills\00", align 1
@.str.5 = private unnamed_addr constant [14 x i8] c"Cookson Hills\00", align 1
@.str.6 = private unnamed_addr constant [9 x i8] c"Harmonic\00", align 1
@.str.7 = private unnamed_addr constant [13 x i8] c"Riemann Zeta\00", align 1
@.str.8 = private unnamed_addr constant [21 x i8] c"Alternating Harmonic\00", align 1
@.str.9 = private unnamed_addr constant [8 x i8] c"Gregory\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <2 x double> @make_vec(double noundef %0, double noundef %1) local_unnamed_addr #0 {
  %3 = insertelement <2 x double> poison, double %0, i64 0
  %4 = insertelement <2 x double> %3, double %1, i64 1
  ret <2 x double> %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef double @sum_vec(<2 x double> noundef %0) local_unnamed_addr #0 {
  %2 = shufflevector <2 x double> %0, <2 x double> poison, <2 x i32> <i32 1, i32 poison>
  %3 = fadd <2 x double> %0, %2
  %4 = extractelement <2 x double> %3, i64 0
  ret double %4
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  br label %3

3:                                                ; preds = %2, %3
  %4 = phi double [ 0.000000e+00, %2 ], [ %13, %3 ]
  %5 = phi double [ 0.000000e+00, %2 ], [ %16, %3 ]
  %6 = phi double [ 0.000000e+00, %2 ], [ %24, %3 ]
  %7 = phi double [ 0.000000e+00, %2 ], [ %28, %3 ]
  %8 = phi i32 [ 1, %2 ], [ %29, %3 ]
  %9 = uitofp nneg i32 %8 to double
  %10 = add nsw i32 %8, -1
  %11 = sitofp i32 %10 to double
  %12 = tail call double @pow(double noundef 0x3FE5555555555555, double noundef %11) #5, !tbaa !6
  %13 = fadd double %4, %12
  %14 = tail call double @sqrt(double noundef %9) #5, !tbaa !6
  %15 = fdiv double 1.000000e+00, %14
  %16 = fadd double %5, %15
  %17 = fmul double %9, %9
  %18 = fmul double %17, %9
  %19 = tail call double @sin(double noundef %9) #5, !tbaa !6
  %20 = tail call double @cos(double noundef %9) #5, !tbaa !6
  %21 = fmul double %18, %19
  %22 = fmul double %19, %21
  %23 = fdiv double 1.000000e+00, %22
  %24 = fadd double %6, %23
  %25 = fmul double %18, %20
  %26 = fmul double %20, %25
  %27 = fdiv double 1.000000e+00, %26
  %28 = fadd double %7, %27
  %29 = add nuw nsw i32 %8, 1
  %30 = icmp eq i32 %29, 2500001
  br i1 %30, label %31, label %3, !llvm.loop !10

31:                                               ; preds = %3, %31
  %32 = phi <2 x double> [ %43, %31 ], [ zeroinitializer, %3 ]
  %33 = phi <2 x double> [ %52, %31 ], [ <double 1.000000e+00, double 2.000000e+00>, %3 ]
  %34 = phi <2 x double> [ %46, %31 ], [ zeroinitializer, %3 ]
  %35 = phi <2 x double> [ %41, %31 ], [ zeroinitializer, %3 ]
  %36 = phi <2 x double> [ %48, %31 ], [ zeroinitializer, %3 ]
  %37 = phi <2 x double> [ %51, %31 ], [ zeroinitializer, %3 ]
  %38 = fadd <2 x double> %33, splat (double 1.000000e+00)
  %39 = fmul <2 x double> %33, %38
  %40 = fdiv <2 x double> splat (double 1.000000e+00), %39
  %41 = fadd <2 x double> %35, %40
  %42 = fdiv <2 x double> splat (double 1.000000e+00), %33
  %43 = fadd <2 x double> %32, %42
  %44 = fmul <2 x double> %33, %33
  %45 = fdiv <2 x double> splat (double 1.000000e+00), %44
  %46 = fadd <2 x double> %34, %45
  %47 = fdiv <2 x double> <double 1.000000e+00, double -1.000000e+00>, %33
  %48 = fadd <2 x double> %36, %47
  %49 = tail call <2 x double> @llvm.fmuladd.v2f64(<2 x double> %33, <2 x double> splat (double 2.000000e+00), <2 x double> splat (double -1.000000e+00))
  %50 = fdiv <2 x double> <double 1.000000e+00, double -1.000000e+00>, %49
  %51 = fadd <2 x double> %37, %50
  %52 = fadd <2 x double> %33, splat (double 2.000000e+00)
  %53 = extractelement <2 x double> %52, i64 0
  %54 = fcmp ugt double %53, 2.500000e+06
  br i1 %54, label %55, label %31, !llvm.loop !12

55:                                               ; preds = %31
  %56 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %13, ptr noundef nonnull @.str.1)
  %57 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %16, ptr noundef nonnull @.str.2)
  %58 = shufflevector <2 x double> %41, <2 x double> poison, <2 x i32> <i32 1, i32 poison>
  %59 = fadd <2 x double> %41, %58
  %60 = extractelement <2 x double> %59, i64 0
  %61 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %60, ptr noundef nonnull @.str.3)
  %62 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %24, ptr noundef nonnull @.str.4)
  %63 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %28, ptr noundef nonnull @.str.5)
  %64 = shufflevector <2 x double> %43, <2 x double> poison, <2 x i32> <i32 1, i32 poison>
  %65 = fadd <2 x double> %43, %64
  %66 = extractelement <2 x double> %65, i64 0
  %67 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %66, ptr noundef nonnull @.str.6)
  %68 = shufflevector <2 x double> %46, <2 x double> poison, <2 x i32> <i32 1, i32 poison>
  %69 = fadd <2 x double> %46, %68
  %70 = extractelement <2 x double> %69, i64 0
  %71 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %70, ptr noundef nonnull @.str.7)
  %72 = shufflevector <2 x double> %48, <2 x double> poison, <2 x i32> <i32 1, i32 poison>
  %73 = fadd <2 x double> %48, %72
  %74 = extractelement <2 x double> %73, i64 0
  %75 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %74, ptr noundef nonnull @.str.8)
  %76 = shufflevector <2 x double> %51, <2 x double> poison, <2 x i32> <i32 1, i32 poison>
  %77 = fadd <2 x double> %51, %76
  %78 = extractelement <2 x double> %77, i64 0
  %79 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %78, ptr noundef nonnull @.str.9)
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @pow(double noundef, double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sqrt(double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sin(double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @cos(double noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x double> @llvm.fmuladd.v2f64(<2 x double>, <2 x double>, <2 x double>) #3

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
