; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/dt.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/dt.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [30 x i8] c" %i iterations of each test. \00", align 1
@.str.1 = private unnamed_addr constant [30 x i8] c" inner loop / array size %i.\0A\00", align 1
@.str.2 = private unnamed_addr constant [4 x i8] c"%f\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #6
  %5 = call i32 @posix_memalign(ptr noundef nonnull %3, i64 noundef 16, i64 noundef 16384) #6
  %6 = call i32 @posix_memalign(ptr noundef nonnull %4, i64 noundef 16, i64 noundef 16384) #6
  %7 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 131072)
  %8 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 2048)
  %9 = load ptr, ptr %3, align 8, !tbaa !6
  %10 = load ptr, ptr %4, align 8, !tbaa !6
  br label %11

11:                                               ; preds = %2, %11
  %12 = phi i64 [ 0, %2 ], [ %24, %11 ]
  %13 = sub nuw nsw i64 2048, %12
  %14 = uitofp nneg i64 %13 to float
  %15 = call float @cosf(float noundef %14) #6, !tbaa !11
  %16 = fpext float %15 to double
  %17 = fmul double %16, 0x3FF000001AD7F29B
  %18 = getelementptr inbounds nuw double, ptr %9, i64 %12
  store double %17, ptr %18, align 8, !tbaa !13
  %19 = uitofp nneg i64 %12 to float
  %20 = call float @sinf(float noundef %19) #6, !tbaa !11
  %21 = fpext float %20 to double
  %22 = call double @llvm.fmuladd.f64(double %21, double 1.000000e-10, double 1.000000e+00)
  %23 = getelementptr inbounds nuw double, ptr %10, i64 %12
  store double %22, ptr %23, align 8, !tbaa !13
  %24 = add nuw nsw i64 %12, 1
  %25 = icmp eq i64 %24, 2048
  br i1 %25, label %26, label %11, !llvm.loop !15

26:                                               ; preds = %11
  call void @llvm.experimental.noalias.scope.decl(metadata !17)
  call void @llvm.experimental.noalias.scope.decl(metadata !20)
  br label %27

27:                                               ; preds = %26, %43
  %28 = phi i64 [ 0, %26 ], [ %44, %43 ]
  br label %29

29:                                               ; preds = %29, %27
  %30 = phi i64 [ 0, %27 ], [ %41, %29 ]
  %31 = getelementptr inbounds nuw double, ptr %10, i64 %30
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 16
  %33 = load <2 x double>, ptr %31, align 8, !tbaa !13, !alias.scope !20, !noalias !17
  %34 = load <2 x double>, ptr %32, align 8, !tbaa !13, !alias.scope !20, !noalias !17
  %35 = getelementptr inbounds nuw double, ptr %9, i64 %30
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 16
  %37 = load <2 x double>, ptr %35, align 8, !tbaa !13, !alias.scope !17, !noalias !20
  %38 = load <2 x double>, ptr %36, align 8, !tbaa !13, !alias.scope !17, !noalias !20
  %39 = fdiv <2 x double> %37, %33
  %40 = fdiv <2 x double> %38, %34
  store <2 x double> %39, ptr %35, align 8, !tbaa !13, !alias.scope !17, !noalias !20
  store <2 x double> %40, ptr %36, align 8, !tbaa !13, !alias.scope !17, !noalias !20
  %41 = add nuw i64 %30, 4
  %42 = icmp eq i64 %41, 2048
  br i1 %42, label %43, label %29, !llvm.loop !22

43:                                               ; preds = %29
  %44 = add nuw nsw i64 %28, 1
  %45 = icmp eq i64 %44, 131072
  br i1 %45, label %46, label %27, !llvm.loop !25

46:                                               ; preds = %43
  %47 = load double, ptr %9, align 8, !tbaa !13
  %48 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, double noundef %47)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #6
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind
declare i32 @posix_memalign(ptr noundef, i64 noundef, i64 noundef) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare float @cosf(float noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(errnomem: write)
declare float @sinf(float noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #5

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(errnomem: write) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }
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
!7 = !{!"p1 double", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"double", !9, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
!17 = !{!18}
!18 = distinct !{!18, !19, !"double_array_divs_variable: argument 0"}
!19 = distinct !{!19, !"double_array_divs_variable"}
!20 = !{!21}
!21 = distinct !{!21, !19, !"double_array_divs_variable: argument 1"}
!22 = distinct !{!22, !16, !23, !24}
!23 = !{!"llvm.loop.isvectorized", i32 1}
!24 = !{!"llvm.loop.unroll.runtime.disable"}
!25 = distinct !{!25, !16}
