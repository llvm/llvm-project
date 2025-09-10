; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/spectral-norm.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/spectral-norm.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [7 x i8] c"%0.9f\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local double @eval_A(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = add nsw i32 %1, %0
  %4 = add nsw i32 %3, 1
  %5 = mul nsw i32 %4, %3
  %6 = sdiv i32 %5, 2
  %7 = add i32 %0, 1
  %8 = add i32 %7, %6
  %9 = sitofp i32 %8 to double
  %10 = fdiv double 1.000000e+00, %9
  ret double %10
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @eval_A_times_u(i32 noundef %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) local_unnamed_addr #1 {
  %4 = icmp sgt i32 %0, 0
  br i1 %4, label %5, label %31

5:                                                ; preds = %3
  %6 = zext nneg i32 %0 to i64
  br label %7

7:                                                ; preds = %5, %29
  %8 = phi i64 [ 0, %5 ], [ %10, %29 ]
  %9 = getelementptr inbounds nuw double, ptr %2, i64 %8
  store double 0.000000e+00, ptr %9, align 8, !tbaa !6
  %10 = add nuw nsw i64 %8, 1
  %11 = add nuw i64 %8, 1
  %12 = trunc nuw nsw i64 %10 to i32
  br label %13

13:                                               ; preds = %7, %13
  %14 = phi i64 [ 0, %7 ], [ %27, %13 ]
  %15 = phi double [ 0.000000e+00, %7 ], [ %26, %13 ]
  %16 = add nuw nsw i64 %14, %8
  %17 = add i64 %14, %11
  %18 = mul i64 %17, %16
  %19 = trunc i64 %18 to i32
  %20 = lshr i32 %19, 1
  %21 = add i32 %20, %12
  %22 = sitofp i32 %21 to double
  %23 = fdiv double 1.000000e+00, %22
  %24 = getelementptr inbounds nuw double, ptr %1, i64 %14
  %25 = load double, ptr %24, align 8, !tbaa !6
  %26 = tail call double @llvm.fmuladd.f64(double %23, double %25, double %15)
  store double %26, ptr %9, align 8, !tbaa !6
  %27 = add nuw nsw i64 %14, 1
  %28 = icmp eq i64 %27, %6
  br i1 %28, label %29, label %13, !llvm.loop !10

29:                                               ; preds = %13
  %30 = icmp eq i64 %10, %6
  br i1 %30, label %31, label %7, !llvm.loop !12

31:                                               ; preds = %29, %3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @eval_At_times_u(i32 noundef %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) local_unnamed_addr #1 {
  %4 = icmp sgt i32 %0, 0
  br i1 %4, label %5, label %31

5:                                                ; preds = %3
  %6 = zext nneg i32 %0 to i64
  br label %7

7:                                                ; preds = %5, %28
  %8 = phi i64 [ 0, %5 ], [ %29, %28 ]
  %9 = getelementptr inbounds nuw double, ptr %2, i64 %8
  store double 0.000000e+00, ptr %9, align 8, !tbaa !6
  %10 = add nuw i64 %8, 1
  br label %11

11:                                               ; preds = %7, %11
  %12 = phi i64 [ 0, %7 ], [ %19, %11 ]
  %13 = phi double [ 0.000000e+00, %7 ], [ %26, %11 ]
  %14 = add nuw nsw i64 %12, %8
  %15 = add i64 %12, %10
  %16 = mul i64 %15, %14
  %17 = trunc i64 %16 to i32
  %18 = lshr i32 %17, 1
  %19 = add nuw nsw i64 %12, 1
  %20 = trunc nuw nsw i64 %19 to i32
  %21 = add nuw i32 %18, %20
  %22 = sitofp i32 %21 to double
  %23 = fdiv double 1.000000e+00, %22
  %24 = getelementptr inbounds nuw double, ptr %1, i64 %12
  %25 = load double, ptr %24, align 8, !tbaa !6
  %26 = tail call double @llvm.fmuladd.f64(double %23, double %25, double %13)
  store double %26, ptr %9, align 8, !tbaa !6
  %27 = icmp eq i64 %19, %6
  br i1 %27, label %28, label %11, !llvm.loop !13

28:                                               ; preds = %11
  %29 = add nuw nsw i64 %8, 1
  %30 = icmp eq i64 %29, %6
  br i1 %30, label %31, label %7, !llvm.loop !14

31:                                               ; preds = %28, %3
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @eval_AtA_times_u(i32 noundef %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) local_unnamed_addr #1 {
  %4 = zext i32 %0 to i64
  %5 = alloca double, i64 %4, align 8
  %6 = icmp sgt i32 %0, 0
  br i1 %6, label %7, label %53

7:                                                ; preds = %3, %28
  %8 = phi i64 [ %10, %28 ], [ 0, %3 ]
  %9 = getelementptr inbounds nuw double, ptr %5, i64 %8
  %10 = add nuw nsw i64 %8, 1
  %11 = trunc nuw nsw i64 %10 to i32
  br label %12

12:                                               ; preds = %12, %7
  %13 = phi i64 [ 0, %7 ], [ %26, %12 ]
  %14 = phi double [ 0.000000e+00, %7 ], [ %25, %12 ]
  %15 = add nuw nsw i64 %13, %8
  %16 = add nuw nsw i64 %13, %10
  %17 = mul i64 %16, %15
  %18 = trunc i64 %17 to i32
  %19 = lshr i32 %18, 1
  %20 = add i32 %19, %11
  %21 = sitofp i32 %20 to double
  %22 = fdiv double 1.000000e+00, %21
  %23 = getelementptr inbounds nuw double, ptr %1, i64 %13
  %24 = load double, ptr %23, align 8, !tbaa !6
  %25 = tail call double @llvm.fmuladd.f64(double %22, double %24, double %14)
  %26 = add nuw nsw i64 %13, 1
  %27 = icmp eq i64 %26, %4
  br i1 %27, label %28, label %12, !llvm.loop !10

28:                                               ; preds = %12
  store double %25, ptr %9, align 8, !tbaa !6
  %29 = icmp eq i64 %10, %4
  br i1 %29, label %30, label %7, !llvm.loop !12

30:                                               ; preds = %28, %51
  %31 = phi i64 [ %33, %51 ], [ 0, %28 ]
  %32 = getelementptr inbounds nuw double, ptr %2, i64 %31
  %33 = add nuw nsw i64 %31, 1
  br label %34

34:                                               ; preds = %34, %30
  %35 = phi i64 [ 0, %30 ], [ %42, %34 ]
  %36 = phi double [ 0.000000e+00, %30 ], [ %49, %34 ]
  %37 = add nuw nsw i64 %35, %31
  %38 = add nuw nsw i64 %35, %33
  %39 = mul i64 %38, %37
  %40 = trunc i64 %39 to i32
  %41 = lshr i32 %40, 1
  %42 = add nuw nsw i64 %35, 1
  %43 = trunc nuw nsw i64 %42 to i32
  %44 = add nuw i32 %41, %43
  %45 = sitofp i32 %44 to double
  %46 = fdiv double 1.000000e+00, %45
  %47 = getelementptr inbounds nuw double, ptr %5, i64 %35
  %48 = load double, ptr %47, align 8, !tbaa !6
  %49 = tail call double @llvm.fmuladd.f64(double %46, double %48, double %36)
  %50 = icmp eq i64 %42, %4
  br i1 %50, label %51, label %34, !llvm.loop !13

51:                                               ; preds = %34
  store double %49, ptr %32, align 8, !tbaa !6
  %52 = icmp eq i64 %33, %4
  br i1 %52, label %53, label %30, !llvm.loop !14

53:                                               ; preds = %51, %3
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #3 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %7, label %4

4:                                                ; preds = %2
  %5 = alloca [2000 x double], align 8
  %6 = alloca [2000 x double], align 8
  br label %16

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !15
  %10 = tail call i64 @strtol(ptr noundef nonnull captures(none) %9, ptr noundef null, i32 noundef 10) #7
  %11 = trunc i64 %10 to i32
  %12 = and i64 %10, 4294967295
  %13 = alloca double, i64 %12, align 8
  %14 = alloca double, i64 %12, align 8
  %15 = icmp sgt i32 %11, 0
  br i1 %15, label %16, label %34

16:                                               ; preds = %4, %7
  %17 = phi ptr [ %6, %4 ], [ %14, %7 ]
  %18 = phi ptr [ %5, %4 ], [ %13, %7 ]
  %19 = phi i64 [ 2000, %4 ], [ %12, %7 ]
  %20 = phi i32 [ 2000, %4 ], [ %11, %7 ]
  %21 = icmp samesign ult i64 %19, 4
  br i1 %21, label %32, label %22

22:                                               ; preds = %16
  %23 = and i64 %19, 4294967292
  br label %24

24:                                               ; preds = %24, %22
  %25 = phi i64 [ 0, %22 ], [ %28, %24 ]
  %26 = getelementptr inbounds nuw double, ptr %18, i64 %25
  %27 = getelementptr inbounds nuw i8, ptr %26, i64 16
  store <2 x double> splat (double 1.000000e+00), ptr %26, align 8, !tbaa !6
  store <2 x double> splat (double 1.000000e+00), ptr %27, align 8, !tbaa !6
  %28 = add nuw i64 %25, 4
  %29 = icmp eq i64 %28, %23
  br i1 %29, label %30, label %24, !llvm.loop !18

30:                                               ; preds = %24
  %31 = icmp eq i64 %19, %23
  br i1 %31, label %34, label %32

32:                                               ; preds = %16, %30
  %33 = phi i64 [ 0, %16 ], [ %23, %30 ]
  br label %73

34:                                               ; preds = %73, %30, %7
  %35 = phi i1 [ false, %7 ], [ true, %30 ], [ true, %73 ]
  %36 = phi ptr [ %14, %7 ], [ %17, %30 ], [ %17, %73 ]
  %37 = phi ptr [ %13, %7 ], [ %18, %30 ], [ %18, %73 ]
  %38 = phi i64 [ %12, %7 ], [ %19, %30 ], [ %19, %73 ]
  %39 = phi i32 [ %11, %7 ], [ %20, %30 ], [ %20, %73 ]
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %37, ptr noundef nonnull %36)
  call void @eval_AtA_times_u(i32 noundef %39, ptr noundef nonnull %36, ptr noundef nonnull %37)
  br i1 %35, label %40, label %94

40:                                               ; preds = %34
  %41 = icmp ult i64 %38, 2
  br i1 %41, label %69, label %42

42:                                               ; preds = %40
  %43 = and i64 %38, -2
  br label %44

44:                                               ; preds = %44, %42
  %45 = phi i64 [ 0, %42 ], [ %63, %44 ]
  %46 = phi <2 x double> [ zeroinitializer, %42 ], [ %62, %44 ]
  %47 = getelementptr inbounds nuw double, ptr %37, i64 %45
  %48 = getelementptr inbounds nuw double, ptr %36, i64 %45
  %49 = getelementptr inbounds nuw double, ptr %36, i64 %45
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 8
  %51 = load double, ptr %48, align 8, !tbaa !6
  %52 = load double, ptr %50, align 8, !tbaa !6
  %53 = load <2 x double>, ptr %47, align 8, !tbaa !6
  %54 = shufflevector <2 x double> %53, <2 x double> poison, <2 x i32> <i32 poison, i32 0>
  %55 = insertelement <2 x double> %54, double %51, i64 0
  %56 = shufflevector <2 x double> %55, <2 x double> poison, <2 x i32> zeroinitializer
  %57 = fmul <2 x double> %55, %56
  %58 = insertelement <2 x double> %53, double %52, i64 0
  %59 = shufflevector <2 x double> %58, <2 x double> poison, <2 x i32> zeroinitializer
  %60 = fmul <2 x double> %58, %59
  %61 = fadd <2 x double> %46, %57
  %62 = fadd <2 x double> %61, %60
  %63 = add nuw i64 %45, 2
  %64 = icmp eq i64 %63, %43
  br i1 %64, label %65, label %44, !llvm.loop !21

65:                                               ; preds = %44
  %66 = icmp eq i64 %38, %43
  %67 = extractelement <2 x double> %62, i64 0
  %68 = extractelement <2 x double> %62, i64 1
  br i1 %66, label %90, label %69

69:                                               ; preds = %40, %65
  %70 = phi i64 [ 0, %40 ], [ %43, %65 ]
  %71 = phi double [ 0.000000e+00, %40 ], [ %67, %65 ]
  %72 = phi double [ 0.000000e+00, %40 ], [ %68, %65 ]
  br label %78

73:                                               ; preds = %32, %73
  %74 = phi i64 [ %76, %73 ], [ %33, %32 ]
  %75 = getelementptr inbounds nuw double, ptr %18, i64 %74
  store double 1.000000e+00, ptr %75, align 8, !tbaa !6
  %76 = add nuw nsw i64 %74, 1
  %77 = icmp eq i64 %76, %19
  br i1 %77, label %34, label %73, !llvm.loop !22

78:                                               ; preds = %69, %78
  %79 = phi i64 [ %88, %78 ], [ %70, %69 ]
  %80 = phi double [ %87, %78 ], [ %71, %69 ]
  %81 = phi double [ %86, %78 ], [ %72, %69 ]
  %82 = getelementptr inbounds nuw double, ptr %37, i64 %79
  %83 = load double, ptr %82, align 8, !tbaa !6
  %84 = getelementptr inbounds nuw double, ptr %36, i64 %79
  %85 = load double, ptr %84, align 8, !tbaa !6
  %86 = tail call double @llvm.fmuladd.f64(double %83, double %85, double %81)
  %87 = tail call double @llvm.fmuladd.f64(double %85, double %85, double %80)
  %88 = add nuw nsw i64 %79, 1
  %89 = icmp eq i64 %88, %38
  br i1 %89, label %90, label %78, !llvm.loop !23

90:                                               ; preds = %78, %65
  %91 = phi double [ %68, %65 ], [ %86, %78 ]
  %92 = phi double [ %67, %65 ], [ %87, %78 ]
  %93 = fdiv double %91, %92
  br label %94

94:                                               ; preds = %90, %34
  %95 = phi double [ 0x7FF8000000000000, %34 ], [ %93, %90 ]
  %96 = tail call double @sqrt(double noundef %95) #7, !tbaa !24
  %97 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %96)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare double @sqrt(double noundef) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nounwind }

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
!12 = distinct !{!12, !11}
!13 = distinct !{!13, !11}
!14 = distinct !{!14, !11}
!15 = !{!16, !16, i64 0}
!16 = !{!"p1 omnipotent char", !17, i64 0}
!17 = !{!"any pointer", !8, i64 0}
!18 = distinct !{!18, !11, !19, !20}
!19 = !{!"llvm.loop.isvectorized", i32 1}
!20 = !{!"llvm.loop.unroll.runtime.disable"}
!21 = distinct !{!21, !11, !19, !20}
!22 = distinct !{!22, !11, !20, !19}
!23 = distinct !{!23, !11, !19}
!24 = !{!25, !25, i64 0}
!25 = !{!"int", !8, i64 0}
