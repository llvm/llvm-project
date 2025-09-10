; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/lib/strnlen.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/lib/strnlen.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local i64 @strnlen(ptr noundef readonly captures(none) %0, i64 noundef %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @inside_main, align 4, !tbaa !6
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %10

5:                                                ; preds = %2
  %6 = load i8, ptr %0, align 1, !tbaa !10
  %7 = icmp eq i8 %6, 0
  %8 = icmp eq i64 %1, 0
  %9 = or i1 %7, %8
  br i1 %9, label %21, label %11

10:                                               ; preds = %2
  tail call void @abort() #2
  unreachable

11:                                               ; preds = %5, %11
  %12 = phi i64 [ %15, %11 ], [ 0, %5 ]
  %13 = phi i64 [ %14, %11 ], [ %1, %5 ]
  %14 = add i64 %13, -1
  %15 = add nuw i64 %12, 1
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 %15
  %17 = load i8, ptr %16, align 1, !tbaa !10
  %18 = icmp eq i8 %17, 0
  %19 = icmp eq i64 %14, 0
  %20 = select i1 %18, i1 true, i1 %19
  br i1 %20, label %21, label %11, !llvm.loop !11

21:                                               ; preds = %11, %5
  %22 = phi i64 [ 0, %5 ], [ %15, %11 ]
  ret i64 %22
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
