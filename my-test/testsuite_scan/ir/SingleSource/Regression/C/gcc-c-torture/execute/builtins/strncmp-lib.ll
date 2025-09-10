; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strncmp-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strncmp-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local range(i32 -255, 256) i32 @strncmp(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = load i32, ptr @inside_main, align 4, !tbaa !6
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %8

6:                                                ; preds = %3
  %7 = icmp eq i64 %2, 0
  br i1 %7, label %27, label %9

8:                                                ; preds = %3
  tail call void @abort() #2
  unreachable

9:                                                ; preds = %6, %22
  %10 = phi ptr [ %23, %22 ], [ %1, %6 ]
  %11 = phi ptr [ %24, %22 ], [ %0, %6 ]
  %12 = phi i64 [ %25, %22 ], [ %2, %6 ]
  %13 = load i8, ptr %11, align 1, !tbaa !10
  %14 = load i8, ptr %10, align 1, !tbaa !10
  %15 = icmp ne i8 %13, 0
  %16 = icmp eq i8 %13, %14
  %17 = select i1 %15, i1 %16, i1 false
  br i1 %17, label %22, label %18

18:                                               ; preds = %9
  %19 = zext i8 %13 to i32
  %20 = zext i8 %14 to i32
  %21 = sub nsw i32 %19, %20
  br label %27

22:                                               ; preds = %9
  %23 = getelementptr inbounds nuw i8, ptr %10, i64 1
  %24 = getelementptr inbounds nuw i8, ptr %11, i64 1
  %25 = add i64 %12, -1
  %26 = icmp eq i64 %25, 0
  br i1 %26, label %27, label %9, !llvm.loop !11

27:                                               ; preds = %22, %6, %18
  %28 = phi i32 [ %21, %18 ], [ 0, %6 ], [ 0, %22 ]
  ret i32 %28
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
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
