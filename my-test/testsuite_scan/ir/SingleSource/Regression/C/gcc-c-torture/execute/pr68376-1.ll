; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68376-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68376-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global i32 1, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i8 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = load i32, ptr @b, align 4
  %3 = load i8, ptr @d, align 4
  %4 = icmp slt i32 %1, 1
  br i1 %4, label %5, label %36

5:                                                ; preds = %0
  %6 = load i32, ptr @c, align 4
  %7 = freeze i32 %6
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %11

9:                                                ; preds = %5
  %10 = icmp slt i32 %2, 1
  br i1 %10, label %16, label %34

11:                                               ; preds = %5, %29
  %12 = phi i32 [ %32, %29 ], [ %1, %5 ]
  %13 = phi i32 [ %31, %29 ], [ %2, %5 ]
  %14 = phi i8 [ %30, %29 ], [ %3, %5 ]
  %15 = icmp slt i32 %13, 1
  br i1 %15, label %20, label %29

16:                                               ; preds = %9
  %17 = icmp sgt i8 %3, 0
  %18 = sext i1 %17 to i8
  %19 = xor i8 %3, %18
  store i8 %19, ptr @d, align 4, !tbaa !10
  tail call void @abort() #2
  unreachable

20:                                               ; preds = %11, %20
  %21 = phi i32 [ %26, %20 ], [ %13, %11 ]
  %22 = phi i8 [ %25, %20 ], [ %14, %11 ]
  %23 = icmp sgt i8 %22, 0
  %24 = sext i1 %23 to i8
  %25 = xor i8 %22, %24
  %26 = add i32 %21, 1
  %27 = icmp eq i32 %21, 0
  br i1 %27, label %28, label %20, !llvm.loop !11

28:                                               ; preds = %20
  store i32 1, ptr @b, align 4, !tbaa !6
  store i8 %25, ptr @d, align 4, !tbaa !10
  br label %29

29:                                               ; preds = %28, %11
  %30 = phi i8 [ %25, %28 ], [ %14, %11 ]
  %31 = phi i32 [ 1, %28 ], [ %13, %11 ]
  %32 = add nsw i32 %12, 1
  %33 = icmp eq i32 %12, 0
  br i1 %33, label %34, label %11, !llvm.loop !13

34:                                               ; preds = %29, %9
  %35 = phi i8 [ %3, %9 ], [ %30, %29 ]
  store i32 1, ptr @a, align 4, !tbaa !6
  br label %36

36:                                               ; preds = %34, %0
  %37 = phi i8 [ %3, %0 ], [ %35, %34 ]
  %38 = icmp eq i8 %37, 0
  br i1 %38, label %40, label %39

39:                                               ; preds = %36
  tail call void @abort() #2
  unreachable

40:                                               ; preds = %36
  ret i32 0
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!8, !8, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12}
