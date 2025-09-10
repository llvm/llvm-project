; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/lib/stpcpy.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/builtins/lib/stpcpy.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@inside_main = external local_unnamed_addr global i32, align 4

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef ptr @stpcpy(ptr noundef writeonly captures(ret: address, provenance) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @inside_main, align 4, !tbaa !6
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %5, label %8

5:                                                ; preds = %2
  %6 = load i8, ptr %1, align 1, !tbaa !10
  %7 = icmp eq i8 %6, 0
  br i1 %7, label %17, label %9

8:                                                ; preds = %2
  tail call void @abort() #2
  unreachable

9:                                                ; preds = %5, %9
  %10 = phi i8 [ %15, %9 ], [ %6, %5 ]
  %11 = phi ptr [ %13, %9 ], [ %1, %5 ]
  %12 = phi ptr [ %14, %9 ], [ %0, %5 ]
  %13 = getelementptr inbounds nuw i8, ptr %11, i64 1
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 1
  store i8 %10, ptr %12, align 1, !tbaa !10
  %15 = load i8, ptr %13, align 1, !tbaa !10
  %16 = icmp eq i8 %15, 0
  br i1 %16, label %17, label %9, !llvm.loop !11

17:                                               ; preds = %9, %5
  %18 = phi ptr [ %0, %5 ], [ %14, %9 ]
  store i8 0, ptr %18, align 1, !tbaa !10
  ret ptr %18
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
