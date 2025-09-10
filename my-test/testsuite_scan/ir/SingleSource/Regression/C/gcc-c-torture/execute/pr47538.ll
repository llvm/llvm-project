; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr47538.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr47538.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { double, double, ptr, i64 }

@__const.main.c = private unnamed_addr constant [4 x double] [double 1.000000e+01, double 2.000000e+01, double 3.000000e+01, double 4.000000e+01], align 8
@__const.main.e = private unnamed_addr constant [4 x double] [double 1.180000e+02, double 1.180000e+02, double 1.180000e+02, double 1.180000e+02], align 8

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @foo(ptr noundef captures(none) initializes((0, 16)) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %4 = load i64, ptr %3, align 8, !tbaa !6
  %5 = add i64 %4, 1
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load double, ptr %6, align 8, !tbaa !14
  %8 = load double, ptr %1, align 8, !tbaa !15
  %9 = fsub double %7, %8
  %10 = fmul double %9, 2.500000e-01
  store double %8, ptr %0, align 8, !tbaa !15
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %7, ptr %11, align 8, !tbaa !14
  %12 = icmp eq i64 %4, 0
  br i1 %12, label %13, label %16

13:                                               ; preds = %2
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %15 = load ptr, ptr %14, align 8, !tbaa !16
  store double 0.000000e+00, ptr %15, align 8, !tbaa !17
  br label %59

16:                                               ; preds = %2
  %17 = icmp eq i64 %5, 2
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %19 = load ptr, ptr %18, align 8, !tbaa !16
  br i1 %17, label %23, label %20

20:                                               ; preds = %16
  %21 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %22 = load ptr, ptr %21, align 8, !tbaa !16
  br label %30

23:                                               ; preds = %16
  %24 = load double, ptr %19, align 8, !tbaa !17
  %25 = fmul double %10, %24
  %26 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %27 = load ptr, ptr %26, align 8, !tbaa !16
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 8
  store double %25, ptr %28, align 8, !tbaa !17
  %29 = fmul double %25, 2.000000e+00
  store double %29, ptr %27, align 8, !tbaa !17
  br label %59

30:                                               ; preds = %20, %30
  %31 = phi i64 [ 1, %20 ], [ %37, %30 ]
  %32 = phi double [ 1.000000e+00, %20 ], [ %46, %30 ]
  %33 = phi double [ 0.000000e+00, %20 ], [ %45, %30 ]
  %34 = getelementptr double, ptr %19, i64 %31
  %35 = getelementptr i8, ptr %34, i64 -8
  %36 = load double, ptr %35, align 8, !tbaa !17
  %37 = add nuw i64 %31, 1
  %38 = getelementptr inbounds nuw double, ptr %19, i64 %37
  %39 = load double, ptr %38, align 8, !tbaa !17
  %40 = fsub double %36, %39
  %41 = fmul double %10, %40
  %42 = uitofp i64 %31 to double
  %43 = fdiv double %41, %42
  %44 = getelementptr inbounds nuw double, ptr %22, i64 %31
  store double %43, ptr %44, align 8, !tbaa !17
  %45 = tail call double @llvm.fmuladd.f64(double %32, double %43, double %33)
  %46 = fneg double %32
  %47 = icmp eq i64 %37, %4
  br i1 %47, label %48, label %30, !llvm.loop !18

48:                                               ; preds = %30
  %49 = getelementptr double, ptr %19, i64 %4
  %50 = getelementptr i8, ptr %49, i64 -8
  %51 = load double, ptr %50, align 8, !tbaa !17
  %52 = fmul double %10, %51
  %53 = uitofp i64 %5 to double
  %54 = fadd double %53, -1.000000e+00
  %55 = fdiv double %52, %54
  %56 = getelementptr inbounds nuw double, ptr %22, i64 %4
  store double %55, ptr %56, align 8, !tbaa !17
  %57 = tail call double @llvm.fmuladd.f64(double %46, double %55, double %45)
  %58 = fmul double %57, 2.000000e+00
  store double %58, ptr %22, align 8, !tbaa !17
  br label %59

59:                                               ; preds = %23, %48, %13
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca %struct.S, align 8
  %2 = alloca %struct.S, align 16
  %3 = alloca [4 x double], align 8
  %4 = alloca [4 x double], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) @__const.main.c, i64 32, i1 false)
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #6
  store <2 x double> <double 1.000000e+01, double 6.000000e+00>, ptr %2, align 16, !tbaa !17
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr %3, ptr %5, align 16, !tbaa !16
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store ptr %4, ptr %6, align 8, !tbaa !16
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store i64 3, ptr %7, align 8, !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) @__const.main.e, i64 32, i1 false)
  call void @foo(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %8 = load <4 x double>, ptr %4, align 8
  %9 = freeze <4 x double> %8
  %10 = fcmp une <4 x double> %9, <double 0.000000e+00, double 2.000000e+01, double 1.000000e+01, double -1.000000e+01>
  %11 = bitcast <4 x i1> %10 to i4
  %12 = icmp eq i4 %11, 0
  br i1 %12, label %14, label %13

13:                                               ; preds = %0
  call void @abort() #7
  unreachable

14:                                               ; preds = %0
  store i64 2, ptr %7, align 8, !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) @__const.main.e, i64 32, i1 false)
  call void @foo(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %15 = load <4 x double>, ptr %4, align 8
  %16 = freeze <4 x double> %15
  %17 = fcmp une <4 x double> %16, <double 6.000000e+01, double 2.000000e+01, double -1.000000e+01, double 1.180000e+02>
  %18 = bitcast <4 x i1> %17 to i4
  %19 = icmp eq i4 %18, 0
  br i1 %19, label %21, label %20

20:                                               ; preds = %14
  call void @abort() #7
  unreachable

21:                                               ; preds = %14
  store i64 1, ptr %7, align 8, !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) @__const.main.e, i64 32, i1 false)
  call void @foo(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %22 = load <4 x double>, ptr %4, align 8
  %23 = freeze <4 x double> %22
  %24 = fcmp une <4 x double> %23, <double -2.000000e+01, double -1.000000e+01, double 1.180000e+02, double 1.180000e+02>
  %25 = bitcast <4 x i1> %24 to i4
  %26 = icmp eq i4 %25, 0
  br i1 %26, label %28, label %27

27:                                               ; preds = %21
  call void @abort() #7
  unreachable

28:                                               ; preds = %21
  store i64 0, ptr %7, align 8, !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %4, ptr noundef nonnull align 8 dereferenceable(32) @__const.main.e, i64 32, i1 false)
  call void @foo(ptr noundef nonnull %1, ptr noundef nonnull %2)
  %29 = load <4 x double>, ptr %4, align 8
  %30 = freeze <4 x double> %29
  %31 = fcmp une <4 x double> %30, <double 0.000000e+00, double 1.180000e+02, double 1.180000e+02, double 1.180000e+02>
  %32 = bitcast <4 x i1> %31 to i4
  %33 = icmp eq i4 %32, 0
  br i1 %33, label %35, label %34

34:                                               ; preds = %28
  call void @abort() #7
  unreachable

35:                                               ; preds = %28
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #6
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #5

attributes #0 = { nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !13, i64 24}
!7 = !{!"S", !8, i64 0, !8, i64 8, !11, i64 16, !13, i64 24}
!8 = !{!"double", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"p1 double", !12, i64 0}
!12 = !{!"any pointer", !9, i64 0}
!13 = !{!"long", !9, i64 0}
!14 = !{!7, !8, i64 8}
!15 = !{!7, !8, i64 0}
!16 = !{!7, !11, i64 16}
!17 = !{!8, !8, i64 0}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
