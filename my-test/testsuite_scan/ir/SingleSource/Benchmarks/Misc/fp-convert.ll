; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/fp-convert.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/fp-convert.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"Total is %g\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: read) uwtable
define dso_local double @loop(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = icmp sgt i64 %2, 0
  br i1 %4, label %5, label %47

5:                                                ; preds = %3
  %6 = icmp ult i64 %2, 8
  br i1 %6, label %32, label %7

7:                                                ; preds = %5
  %8 = and i64 %2, 9223372036854775800
  br label %9

9:                                                ; preds = %9, %7
  %10 = phi i64 [ 0, %7 ], [ %28, %9 ]
  %11 = phi double [ 0.000000e+00, %7 ], [ %27, %9 ]
  %12 = getelementptr inbounds nuw float, ptr %0, i64 %10
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %14 = load <4 x float>, ptr %12, align 4, !tbaa !6
  %15 = load <4 x float>, ptr %13, align 4, !tbaa !6
  %16 = fpext <4 x float> %14 to <4 x double>
  %17 = fpext <4 x float> %15 to <4 x double>
  %18 = getelementptr inbounds nuw float, ptr %1, i64 %10
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x float>, ptr %18, align 4, !tbaa !6
  %21 = load <4 x float>, ptr %19, align 4, !tbaa !6
  %22 = fpext <4 x float> %20 to <4 x double>
  %23 = fpext <4 x float> %21 to <4 x double>
  %24 = fmul <4 x double> %16, %22
  %25 = fmul <4 x double> %17, %23
  %26 = tail call double @llvm.vector.reduce.fadd.v4f64(double %11, <4 x double> %24)
  %27 = tail call double @llvm.vector.reduce.fadd.v4f64(double %26, <4 x double> %25)
  %28 = add nuw i64 %10, 8
  %29 = icmp eq i64 %28, %8
  br i1 %29, label %30, label %9, !llvm.loop !10

30:                                               ; preds = %9
  %31 = icmp eq i64 %2, %8
  br i1 %31, label %47, label %32

32:                                               ; preds = %5, %30
  %33 = phi double [ 0.000000e+00, %5 ], [ %27, %30 ]
  %34 = phi i64 [ 0, %5 ], [ %8, %30 ]
  br label %35

35:                                               ; preds = %32, %35
  %36 = phi double [ %44, %35 ], [ %33, %32 ]
  %37 = phi i64 [ %45, %35 ], [ %34, %32 ]
  %38 = getelementptr inbounds nuw float, ptr %0, i64 %37
  %39 = load float, ptr %38, align 4, !tbaa !6
  %40 = fpext float %39 to double
  %41 = getelementptr inbounds nuw float, ptr %1, i64 %37
  %42 = load float, ptr %41, align 4, !tbaa !6
  %43 = fpext float %42 to double
  %44 = tail call double @llvm.fmuladd.f64(double %40, double %43, double %36)
  %45 = add nuw nsw i64 %37, 1
  %46 = icmp eq i64 %45, %2
  br i1 %46, label %47, label %35, !llvm.loop !14

47:                                               ; preds = %35, %30, %3
  %48 = phi double [ 0.000000e+00, %3 ], [ %27, %30 ], [ %44, %35 ]
  ret double %48
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #3 {
  %3 = alloca [2048 x float], align 4
  %4 = alloca [2048 x float], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #6
  br label %5

5:                                                ; preds = %2, %58
  %6 = phi float [ 1.000000e+00, %2 ], [ %15, %58 ]
  %7 = phi float [ 0.000000e+00, %2 ], [ %14, %58 ]
  %8 = phi double [ 0.000000e+00, %2 ], [ %59, %58 ]
  %9 = phi i32 [ 0, %2 ], [ %60, %58 ]
  %10 = urem i32 %9, 10
  %11 = icmp eq i32 %10, 0
  %12 = fadd float %7, 0x3FB99999A0000000
  %13 = fadd float %6, 0x3FC99999A0000000
  %14 = select i1 %11, float %12, float 0.000000e+00
  %15 = select i1 %11, float %13, float 1.000000e+00
  %16 = insertelement <4 x float> poison, float %14, i64 0
  %17 = shufflevector <4 x float> %16, <4 x float> poison, <4 x i32> zeroinitializer
  %18 = insertelement <4 x float> poison, float %15, i64 0
  %19 = shufflevector <4 x float> %18, <4 x float> poison, <4 x i32> zeroinitializer
  br label %20

20:                                               ; preds = %20, %5
  %21 = phi i64 [ 0, %5 ], [ %34, %20 ]
  %22 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %5 ], [ %35, %20 ]
  %23 = add <4 x i32> %22, splat (i32 4)
  %24 = uitofp nneg <4 x i32> %22 to <4 x float>
  %25 = uitofp nneg <4 x i32> %23 to <4 x float>
  %26 = fadd <4 x float> %17, %24
  %27 = fadd <4 x float> %17, %25
  %28 = getelementptr inbounds nuw float, ptr %3, i64 %21
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 16
  store <4 x float> %26, ptr %28, align 4, !tbaa !6
  store <4 x float> %27, ptr %29, align 4, !tbaa !6
  %30 = fadd <4 x float> %19, %24
  %31 = fadd <4 x float> %19, %25
  %32 = getelementptr inbounds nuw float, ptr %4, i64 %21
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 16
  store <4 x float> %30, ptr %32, align 4, !tbaa !6
  store <4 x float> %31, ptr %33, align 4, !tbaa !6
  %34 = add nuw i64 %21, 8
  %35 = add <4 x i32> %22, splat (i32 8)
  %36 = icmp eq i64 %34, 2048
  br i1 %36, label %37, label %20, !llvm.loop !15

37:                                               ; preds = %20, %37
  %38 = phi i64 [ %56, %37 ], [ 0, %20 ]
  %39 = phi double [ %55, %37 ], [ 0.000000e+00, %20 ]
  %40 = getelementptr inbounds nuw float, ptr %3, i64 %38
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 16
  %42 = load <4 x float>, ptr %40, align 4, !tbaa !6
  %43 = load <4 x float>, ptr %41, align 4, !tbaa !6
  %44 = fpext <4 x float> %42 to <4 x double>
  %45 = fpext <4 x float> %43 to <4 x double>
  %46 = getelementptr inbounds nuw float, ptr %4, i64 %38
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 16
  %48 = load <4 x float>, ptr %46, align 4, !tbaa !6
  %49 = load <4 x float>, ptr %47, align 4, !tbaa !6
  %50 = fpext <4 x float> %48 to <4 x double>
  %51 = fpext <4 x float> %49 to <4 x double>
  %52 = fmul <4 x double> %44, %50
  %53 = fmul <4 x double> %45, %51
  %54 = tail call double @llvm.vector.reduce.fadd.v4f64(double %39, <4 x double> %52)
  %55 = tail call double @llvm.vector.reduce.fadd.v4f64(double %54, <4 x double> %53)
  %56 = add nuw i64 %38, 8
  %57 = icmp eq i64 %56, 2048
  br i1 %57, label %58, label %37, !llvm.loop !16

58:                                               ; preds = %37
  %59 = fadd double %8, %55
  %60 = add nuw nsw i32 %9, 1
  %61 = icmp eq i32 %60, 500000
  br i1 %61, label %62, label %5, !llvm.loop !17

62:                                               ; preds = %58
  %63 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, double noundef %59)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #6
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.vector.reduce.fadd.v4f64(double, <4 x double>) #5

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !13, !12}
!15 = distinct !{!15, !11, !12, !13}
!16 = distinct !{!16, !11, !12, !13}
!17 = distinct !{!17, !11}
