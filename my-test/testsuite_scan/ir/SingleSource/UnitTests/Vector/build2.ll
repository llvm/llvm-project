; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vector/build2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vector/build2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"%f %f %f %f\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test0001(float noundef %0) local_unnamed_addr #0 {
  %2 = insertelement <4 x float> <float poison, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>, float %0, i64 0
  ret <4 x float> %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test0010(float noundef %0) local_unnamed_addr #0 {
  %2 = insertelement <4 x float> <float 0.000000e+00, float poison, float 0.000000e+00, float 0.000000e+00>, float %0, i64 1
  ret <4 x float> %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test0100(float noundef %0) local_unnamed_addr #0 {
  %2 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float poison, float 0.000000e+00>, float %0, i64 2
  ret <4 x float> %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test1000(float noundef %0) local_unnamed_addr #0 {
  %2 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float poison>, float %0, i64 3
  ret <4 x float> %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test0011(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = insertelement <4 x float> <float poison, float poison, float 0.000000e+00, float 0.000000e+00>, float %0, i64 0
  %4 = insertelement <4 x float> %3, float %1, i64 1
  ret <4 x float> %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test0101(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = insertelement <4 x float> <float poison, float 0.000000e+00, float poison, float 0.000000e+00>, float %0, i64 0
  %4 = insertelement <4 x float> %3, float %1, i64 2
  ret <4 x float> %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test1001(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = insertelement <4 x float> <float poison, float 0.000000e+00, float 0.000000e+00, float poison>, float %0, i64 0
  %4 = insertelement <4 x float> %3, float %1, i64 3
  ret <4 x float> %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test0110(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = insertelement <4 x float> <float 0.000000e+00, float poison, float poison, float 0.000000e+00>, float %0, i64 1
  %4 = insertelement <4 x float> %3, float %1, i64 2
  ret <4 x float> %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test1010(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = insertelement <4 x float> <float 0.000000e+00, float poison, float 0.000000e+00, float poison>, float %0, i64 1
  %4 = insertelement <4 x float> %3, float %1, i64 3
  ret <4 x float> %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test1100(float noundef %0, float noundef %1) local_unnamed_addr #0 {
  %3 = insertelement <4 x float> <float 0.000000e+00, float 0.000000e+00, float poison, float poison>, float %0, i64 2
  %4 = insertelement <4 x float> %3, float %1, i64 3
  ret <4 x float> %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test0111(float noundef %0, float noundef %1, float noundef %2) local_unnamed_addr #0 {
  %4 = insertelement <4 x float> <float poison, float poison, float poison, float 0.000000e+00>, float %0, i64 0
  %5 = insertelement <4 x float> %4, float %1, i64 1
  %6 = insertelement <4 x float> %5, float %2, i64 2
  ret <4 x float> %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test1011(float noundef %0, float noundef %1, float noundef %2) local_unnamed_addr #0 {
  %4 = insertelement <4 x float> <float poison, float poison, float 0.000000e+00, float poison>, float %0, i64 0
  %5 = insertelement <4 x float> %4, float %1, i64 1
  %6 = insertelement <4 x float> %5, float %2, i64 3
  ret <4 x float> %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test1101(float noundef %0, float noundef %1, float noundef %2) local_unnamed_addr #0 {
  %4 = insertelement <4 x float> <float poison, float 0.000000e+00, float poison, float poison>, float %0, i64 0
  %5 = insertelement <4 x float> %4, float %1, i64 2
  %6 = insertelement <4 x float> %5, float %2, i64 3
  ret <4 x float> %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local <4 x float> @test1110(float noundef %0, float noundef %1, float noundef %2) local_unnamed_addr #0 {
  %4 = insertelement <4 x float> <float 0.000000e+00, float poison, float poison, float poison>, float %0, i64 1
  %5 = insertelement <4 x float> %4, float %1, i64 2
  %6 = insertelement <4 x float> %5, float %2, i64 3
  ret <4 x float> %6
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  br label %3

3:                                                ; preds = %2, %26
  %4 = phi i32 [ 0, %2 ], [ %27, %26 ]
  br label %5

5:                                                ; preds = %3, %5
  %6 = phi i32 [ 0, %3 ], [ %24, %5 ]
  %7 = phi <4 x float> [ zeroinitializer, %3 ], [ %23, %5 ]
  %8 = phi <4 x float> [ zeroinitializer, %3 ], [ %19, %5 ]
  %9 = phi <4 x float> [ zeroinitializer, %3 ], [ %13, %5 ]
  %10 = fadd <4 x float> %9, <float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00>
  %11 = fadd <4 x float> %10, <float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float 0.000000e+00>
  %12 = fadd <4 x float> %11, <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, float 0.000000e+00>
  %13 = fadd <4 x float> %12, <float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 1.000000e+00>
  %14 = fadd <4 x float> %8, <float 1.000000e+00, float 2.000000e+00, float 0.000000e+00, float 0.000000e+00>
  %15 = fadd <4 x float> %14, <float 1.000000e+00, float 0.000000e+00, float 2.000000e+00, float 0.000000e+00>
  %16 = fadd <4 x float> %15, <float 1.000000e+00, float 0.000000e+00, float 0.000000e+00, float 2.000000e+00>
  %17 = fadd <4 x float> %16, <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 0.000000e+00>
  %18 = fadd <4 x float> %17, <float 0.000000e+00, float 1.000000e+00, float 0.000000e+00, float 2.000000e+00>
  %19 = fadd <4 x float> %18, <float 0.000000e+00, float 0.000000e+00, float 1.000000e+00, float 2.000000e+00>
  %20 = fadd <4 x float> %7, <float 2.000000e+00, float 3.000000e+00, float 1.000000e+00, float 0.000000e+00>
  %21 = fadd <4 x float> %20, <float 1.000000e+00, float 1.000000e+00, float 0.000000e+00, float 2.000000e+00>
  %22 = fadd <4 x float> %21, <float 3.000000e+00, float 0.000000e+00, float 2.000000e+00, float 4.000000e+00>
  %23 = fadd <4 x float> %22, <float 0.000000e+00, float 4.000000e+00, float 6.000000e+00, float 1.000000e+00>
  %24 = add nuw nsw i32 %6, 1
  %25 = icmp eq i32 %24, 2000000
  br i1 %25, label %26, label %5, !llvm.loop !6

26:                                               ; preds = %5
  %27 = add nuw nsw i32 %4, 1
  %28 = icmp eq i32 %27, 100
  br i1 %28, label %29, label %3, !llvm.loop !8

29:                                               ; preds = %26
  %30 = extractelement <4 x float> %13, i64 0
  %31 = fpext float %30 to double
  %32 = extractelement <4 x float> %13, i64 1
  %33 = fpext float %32 to double
  %34 = extractelement <4 x float> %13, i64 2
  %35 = fpext float %34 to double
  %36 = extractelement <4 x float> %13, i64 3
  %37 = fpext float %36 to double
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %31, double noundef %33, double noundef %35, double noundef %37)
  %39 = extractelement <4 x float> %19, i64 0
  %40 = fpext float %39 to double
  %41 = extractelement <4 x float> %19, i64 1
  %42 = fpext float %41 to double
  %43 = extractelement <4 x float> %19, i64 2
  %44 = fpext float %43 to double
  %45 = extractelement <4 x float> %19, i64 3
  %46 = fpext float %45 to double
  %47 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %40, double noundef %42, double noundef %44, double noundef %46)
  %48 = extractelement <4 x float> %23, i64 0
  %49 = fpext float %48 to double
  %50 = extractelement <4 x float> %23, i64 1
  %51 = fpext float %50 to double
  %52 = extractelement <4 x float> %23, i64 2
  %53 = fpext float %52 to double
  %54 = extractelement <4 x float> %23, i64 3
  %55 = fpext float %54 to double
  %56 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %49, double noundef %51, double noundef %53, double noundef %55)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
