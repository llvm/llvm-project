; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memchr-lib.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/memchr-lib.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef ptr @memchr(ptr noundef readonly captures(ret: address, provenance) %0, i32 noundef %1, i64 noundef %2) local_unnamed_addr #0 {
  %4 = load i32, ptr @inside_main, align 4, !tbaa !6
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %10

6:                                                ; preds = %3
  %7 = icmp eq i64 %2, 0
  br i1 %7, label %20, label %8

8:                                                ; preds = %6
  %9 = trunc i32 %1 to i8
  br label %11

10:                                               ; preds = %3
  tail call void @abort() #2
  unreachable

11:                                               ; preds = %8, %16
  %12 = phi ptr [ %0, %8 ], [ %17, %16 ]
  %13 = phi i64 [ %2, %8 ], [ %18, %16 ]
  %14 = load i8, ptr %12, align 1, !tbaa !10
  %15 = icmp eq i8 %14, %9
  br i1 %15, label %20, label %16

16:                                               ; preds = %11
  %17 = getelementptr inbounds nuw i8, ptr %12, i64 1
  %18 = add i64 %13, -1
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %20, label %11, !llvm.loop !11

20:                                               ; preds = %11, %16, %6
  %21 = phi ptr [ null, %6 ], [ null, %16 ], [ %12, %11 ]
  ret ptr %21
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
