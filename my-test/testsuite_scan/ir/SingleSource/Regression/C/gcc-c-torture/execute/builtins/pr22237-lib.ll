; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/pr22237-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/pr22237-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree nounwind uwtable
define dso_local noundef ptr @memcpy(ptr noundef returned writeonly captures(address, ret: address, provenance) %0, ptr noundef readonly captures(address) %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = ptrtoint ptr %1 to i64
  %5 = ptrtoint ptr %0 to i64
  %6 = icmp ult ptr %0, %1
  br i1 %6, label %7, label %11

7:                                                ; preds = %3
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 %2
  %9 = icmp ugt ptr %8, %1
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  tail call void @abort() #2
  unreachable

11:                                               ; preds = %3
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 %2
  %13 = icmp ugt ptr %12, %0
  br i1 %13, label %14, label %15

14:                                               ; preds = %11
  tail call void @abort() #2
  unreachable

15:                                               ; preds = %11, %7
  %16 = icmp eq i64 %2, 0
  br i1 %16, label %72, label %17

17:                                               ; preds = %15
  %18 = icmp ult i64 %2, 8
  %19 = sub i64 %5, %4
  %20 = icmp ult i64 %19, 32
  %21 = or i1 %18, %20
  br i1 %21, label %59, label %22

22:                                               ; preds = %17
  %23 = icmp ult i64 %2, 32
  br i1 %23, label %44, label %24

24:                                               ; preds = %22
  %25 = and i64 %2, -32
  br label %26

26:                                               ; preds = %26, %24
  %27 = phi i64 [ 0, %24 ], [ %34, %26 ]
  %28 = getelementptr i8, ptr %0, i64 %27
  %29 = getelementptr i8, ptr %1, i64 %27
  %30 = getelementptr i8, ptr %29, i64 16
  %31 = load <16 x i8>, ptr %29, align 1, !tbaa !6
  %32 = load <16 x i8>, ptr %30, align 1, !tbaa !6
  %33 = getelementptr i8, ptr %28, i64 16
  store <16 x i8> %31, ptr %28, align 1, !tbaa !6
  store <16 x i8> %32, ptr %33, align 1, !tbaa !6
  %34 = add nuw i64 %27, 32
  %35 = icmp eq i64 %34, %25
  br i1 %35, label %36, label %26, !llvm.loop !9

36:                                               ; preds = %26
  %37 = icmp eq i64 %2, %25
  br i1 %37, label %72, label %38

38:                                               ; preds = %36
  %39 = getelementptr i8, ptr %0, i64 %25
  %40 = getelementptr i8, ptr %1, i64 %25
  %41 = and i64 %2, 31
  %42 = and i64 %2, 24
  %43 = icmp eq i64 %42, 0
  br i1 %43, label %59, label %44

44:                                               ; preds = %38, %22
  %45 = phi i64 [ %25, %38 ], [ 0, %22 ]
  %46 = and i64 %2, -8
  %47 = getelementptr i8, ptr %0, i64 %46
  %48 = getelementptr i8, ptr %1, i64 %46
  %49 = and i64 %2, 7
  br label %50

50:                                               ; preds = %50, %44
  %51 = phi i64 [ %45, %44 ], [ %55, %50 ]
  %52 = getelementptr i8, ptr %0, i64 %51
  %53 = getelementptr i8, ptr %1, i64 %51
  %54 = load <8 x i8>, ptr %53, align 1, !tbaa !6
  store <8 x i8> %54, ptr %52, align 1, !tbaa !6
  %55 = add nuw i64 %51, 8
  %56 = icmp eq i64 %55, %46
  br i1 %56, label %57, label %50, !llvm.loop !13

57:                                               ; preds = %50
  %58 = icmp eq i64 %2, %46
  br i1 %58, label %72, label %59

59:                                               ; preds = %38, %57, %17
  %60 = phi ptr [ %0, %17 ], [ %39, %38 ], [ %47, %57 ]
  %61 = phi ptr [ %1, %17 ], [ %40, %38 ], [ %48, %57 ]
  %62 = phi i64 [ %2, %17 ], [ %41, %38 ], [ %49, %57 ]
  br label %63

63:                                               ; preds = %59, %63
  %64 = phi ptr [ %70, %63 ], [ %60, %59 ]
  %65 = phi ptr [ %68, %63 ], [ %61, %59 ]
  %66 = phi i64 [ %67, %63 ], [ %62, %59 ]
  %67 = add i64 %66, -1
  %68 = getelementptr inbounds nuw i8, ptr %65, i64 1
  %69 = load i8, ptr %65, align 1, !tbaa !6
  %70 = getelementptr inbounds nuw i8, ptr %64, i64 1
  store i8 %69, ptr %64, align 1, !tbaa !6
  %71 = icmp eq i64 %67, 0
  br i1 %71, label %72, label %63, !llvm.loop !14

72:                                               ; preds = %63, %36, %57, %15
  ret ptr %0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10, !11, !12}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!"llvm.loop.isvectorized", i32 1}
!12 = !{!"llvm.loop.unroll.runtime.disable"}
!13 = distinct !{!13, !10, !11, !12}
!14 = distinct !{!14, !10, !11}
