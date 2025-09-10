; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/mempcpy-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/mempcpy-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local ptr @mempcpy(ptr noundef writeonly captures(ret: address, provenance) %0, ptr noundef readonly captures(none) %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = load i32, ptr @inside_main, align 4, !tbaa !6
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %36

8:                                                ; preds = %3
  %9 = icmp eq i64 %2, 0
  br i1 %9, label %46, label %10

10:                                               ; preds = %8
  %11 = icmp ult i64 %2, 32
  %12 = sub i64 %5, %4
  %13 = icmp ult i64 %12, 32
  %14 = or i1 %11, %13
  br i1 %14, label %32, label %15

15:                                               ; preds = %10
  %16 = and i64 %2, -32
  %17 = getelementptr i8, ptr %0, i64 %16
  %18 = getelementptr i8, ptr %1, i64 %16
  %19 = and i64 %2, 31
  br label %20

20:                                               ; preds = %20, %15
  %21 = phi i64 [ 0, %15 ], [ %28, %20 ]
  %22 = getelementptr i8, ptr %0, i64 %21
  %23 = getelementptr i8, ptr %1, i64 %21
  %24 = getelementptr i8, ptr %23, i64 16
  %25 = load <16 x i8>, ptr %23, align 1, !tbaa !10
  %26 = load <16 x i8>, ptr %24, align 1, !tbaa !10
  %27 = getelementptr i8, ptr %22, i64 16
  store <16 x i8> %25, ptr %22, align 1, !tbaa !10
  store <16 x i8> %26, ptr %27, align 1, !tbaa !10
  %28 = add nuw i64 %21, 32
  %29 = icmp eq i64 %28, %16
  br i1 %29, label %30, label %20, !llvm.loop !11

30:                                               ; preds = %20
  %31 = icmp eq i64 %2, %16
  br i1 %31, label %46, label %32

32:                                               ; preds = %10, %30
  %33 = phi ptr [ %0, %10 ], [ %17, %30 ]
  %34 = phi ptr [ %1, %10 ], [ %18, %30 ]
  %35 = phi i64 [ %2, %10 ], [ %19, %30 ]
  br label %37

36:                                               ; preds = %3
  tail call void @abort() #2
  unreachable

37:                                               ; preds = %32, %37
  %38 = phi ptr [ %44, %37 ], [ %33, %32 ]
  %39 = phi ptr [ %42, %37 ], [ %34, %32 ]
  %40 = phi i64 [ %41, %37 ], [ %35, %32 ]
  %41 = add i64 %40, -1
  %42 = getelementptr inbounds nuw i8, ptr %39, i64 1
  %43 = load i8, ptr %39, align 1, !tbaa !10
  %44 = getelementptr inbounds nuw i8, ptr %38, i64 1
  store i8 %43, ptr %38, align 1, !tbaa !10
  %45 = icmp eq i64 %41, 0
  br i1 %45, label %46, label %37, !llvm.loop !15

46:                                               ; preds = %37, %30, %8
  %47 = phi ptr [ %0, %8 ], [ %17, %30 ], [ %44, %37 ]
  ret ptr %47
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind }

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
!10 = !{!8, !8, i64 0}
!11 = distinct !{!11, !12, !13, !14}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
!15 = distinct !{!15, !12, !13}
