; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-05-02-DependentPHI.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-05-02-DependentPHI.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@Node0 = dso_local global { ptr, i32, [4 x i8] } { ptr null, i32 5, [4 x i8] zeroinitializer }, align 8
@Node1 = dso_local global { ptr, i32, [4 x i8] } { ptr @Node0, i32 4, [4 x i8] zeroinitializer }, align 8
@Node2 = dso_local global { ptr, i32, [4 x i8] } { ptr @Node1, i32 3, [4 x i8] zeroinitializer }, align 8
@Node3 = dso_local global { ptr, i32, [4 x i8] } { ptr @Node2, i32 2, [4 x i8] zeroinitializer }, align 8
@Node4 = dso_local global { ptr, i32, [4 x i8] } { ptr @Node3, i32 1, [4 x i8] zeroinitializer }, align 8
@Node5 = dso_local global { ptr, i32, [4 x i8] } { ptr @Node4, i32 0, [4 x i8] zeroinitializer }, align 8
@.str = private unnamed_addr constant [7 x i8] c"%d %d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %10
  %2 = phi ptr [ @Node5, %0 ], [ %13, %10 ]
  %3 = phi ptr [ null, %0 ], [ %2, %10 ]
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %5 = load i32, ptr %4, align 8, !tbaa !6
  %6 = icmp eq ptr %3, null
  br i1 %6, label %10, label %7

7:                                                ; preds = %1
  %8 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %9 = load i32, ptr %8, align 8, !tbaa !6
  br label %10

10:                                               ; preds = %1, %7
  %11 = phi i32 [ %9, %7 ], [ -1, %1 ]
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %5, i32 noundef %11)
  %13 = load ptr, ptr %2, align 8, !tbaa !13
  %14 = icmp eq ptr %13, null
  br i1 %14, label %15, label %1, !llvm.loop !14

15:                                               ; preds = %10
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !12, i64 8}
!7 = !{!"List", !8, i64 0, !12, i64 8}
!8 = !{!"p1 _ZTS4List", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!"int", !10, i64 0}
!13 = !{!7, !8, i64 0}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.mustprogress"}
