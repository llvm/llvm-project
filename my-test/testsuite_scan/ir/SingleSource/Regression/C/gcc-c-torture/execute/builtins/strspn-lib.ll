; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strspn-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/strspn-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef i64 @strcspn(ptr noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @inside_main, align 4, !tbaa !6
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %11

5:                                                ; preds = %2
  %6 = load i8, ptr %0, align 1, !tbaa !10
  %7 = icmp eq i8 %6, 0
  br i1 %7, label %27, label %8

8:                                                ; preds = %5
  %9 = load i8, ptr %1, align 1, !tbaa !10
  %10 = icmp eq i8 %9, 0
  br i1 %10, label %27, label %12

11:                                               ; preds = %2
  tail call void @abort() #2
  unreachable

12:                                               ; preds = %8, %23
  %13 = phi i8 [ %25, %23 ], [ %6, %8 ]
  %14 = phi ptr [ %24, %23 ], [ %0, %8 ]
  br label %19

15:                                               ; preds = %19
  %16 = getelementptr inbounds nuw i8, ptr %21, i64 1
  %17 = load i8, ptr %16, align 1, !tbaa !10
  %18 = icmp eq i8 %17, 0
  br i1 %18, label %27, label %19, !llvm.loop !11

19:                                               ; preds = %12, %15
  %20 = phi i8 [ %9, %12 ], [ %17, %15 ]
  %21 = phi ptr [ %1, %12 ], [ %16, %15 ]
  %22 = icmp eq i8 %13, %20
  br i1 %22, label %23, label %15

23:                                               ; preds = %19
  %24 = getelementptr inbounds nuw i8, ptr %14, i64 1
  %25 = load i8, ptr %24, align 1, !tbaa !10
  %26 = icmp eq i8 %25, 0
  br i1 %26, label %27, label %12, !llvm.loop !13

27:                                               ; preds = %23, %15, %5, %8
  %28 = phi ptr [ %0, %8 ], [ %0, %5 ], [ %14, %15 ], [ %24, %23 ]
  %29 = ptrtoint ptr %28 to i64
  %30 = ptrtoint ptr %0 to i64
  %31 = sub i64 %29, %30
  ret i64 %31
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
!13 = distinct !{!13, !12}
