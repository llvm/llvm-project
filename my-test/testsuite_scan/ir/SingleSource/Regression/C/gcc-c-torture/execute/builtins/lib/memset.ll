; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/lib/memset.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/lib/memset.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: write) uwtable
define dso_local noundef ptr @memset(ptr noundef returned writeonly captures(ret: address, provenance) %0, i32 noundef %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = icmp eq i64 %2, 0
  br i1 %4, label %52, label %5

5:                                                ; preds = %3
  %6 = trunc i32 %1 to i8
  %7 = icmp ult i64 %2, 8
  br i1 %7, label %45, label %8

8:                                                ; preds = %5
  %9 = icmp ult i64 %2, 32
  br i1 %9, label %29, label %10

10:                                               ; preds = %8
  %11 = and i64 %2, -32
  %12 = insertelement <16 x i8> poison, i8 %6, i64 0
  %13 = shufflevector <16 x i8> %12, <16 x i8> poison, <16 x i32> zeroinitializer
  %14 = getelementptr i8, ptr %0, i64 %2
  br label %15

15:                                               ; preds = %15, %10
  %16 = phi i64 [ 0, %10 ], [ %21, %15 ]
  %17 = xor i64 %16, -1
  %18 = getelementptr i8, ptr %14, i64 %17
  %19 = getelementptr inbounds i8, ptr %18, i64 -15
  %20 = getelementptr inbounds i8, ptr %18, i64 -31
  store <16 x i8> %13, ptr %19, align 1, !tbaa !6
  store <16 x i8> %13, ptr %20, align 1, !tbaa !6
  %21 = add nuw i64 %16, 32
  %22 = icmp eq i64 %21, %11
  br i1 %22, label %23, label %15, !llvm.loop !9

23:                                               ; preds = %15
  %24 = icmp eq i64 %2, %11
  br i1 %24, label %52, label %25

25:                                               ; preds = %23
  %26 = and i64 %2, 31
  %27 = and i64 %2, 24
  %28 = icmp eq i64 %27, 0
  br i1 %28, label %45, label %29

29:                                               ; preds = %25, %8
  %30 = phi i64 [ %11, %25 ], [ 0, %8 ]
  %31 = and i64 %2, -8
  %32 = and i64 %2, 7
  %33 = insertelement <8 x i8> poison, i8 %6, i64 0
  %34 = shufflevector <8 x i8> %33, <8 x i8> poison, <8 x i32> zeroinitializer
  %35 = getelementptr i8, ptr %0, i64 %2
  br label %36

36:                                               ; preds = %36, %29
  %37 = phi i64 [ %30, %29 ], [ %41, %36 ]
  %38 = xor i64 %37, -1
  %39 = getelementptr i8, ptr %35, i64 %38
  %40 = getelementptr inbounds i8, ptr %39, i64 -7
  store <8 x i8> %34, ptr %40, align 1, !tbaa !6
  %41 = add nuw i64 %37, 8
  %42 = icmp eq i64 %41, %31
  br i1 %42, label %43, label %36, !llvm.loop !13

43:                                               ; preds = %36
  %44 = icmp eq i64 %2, %31
  br i1 %44, label %52, label %45

45:                                               ; preds = %25, %43, %5
  %46 = phi i64 [ %2, %5 ], [ %26, %25 ], [ %32, %43 ]
  br label %47

47:                                               ; preds = %45, %47
  %48 = phi i64 [ %49, %47 ], [ %46, %45 ]
  %49 = add i64 %48, -1
  %50 = getelementptr inbounds nuw i8, ptr %0, i64 %49
  store i8 %6, ptr %50, align 1, !tbaa !6
  %51 = icmp eq i64 %49, 0
  br i1 %51, label %52, label %47, !llvm.loop !14

52:                                               ; preds = %47, %23, %43, %3
  ret ptr %0
}

attributes #0 = { nofree noinline norecurse nosync nounwind memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.isvectorized", i32 1}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = distinct !{!13, !10, !11, !12}
!14 = distinct !{!14, !10, !12, !11}
