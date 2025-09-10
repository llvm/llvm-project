; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/testcase-Value-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/testcase-Value-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i16 1, align 2
@d = dso_local local_unnamed_addr global i16 5, align 2
@h = dso_local local_unnamed_addr global i16 1, align 2
@e = dso_local local_unnamed_addr global i32 1, align 4
@f = dso_local local_unnamed_addr global i32 20, align 4
@g = dso_local local_unnamed_addr global i32 1, align 4
@j = dso_local local_unnamed_addr global i32 1, align 4
@c = dso_local local_unnamed_addr global [6 x i8] zeroinitializer, align 1
@a = dso_local local_unnamed_addr global i8 0, align 4
@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @f, align 4, !tbaa !6
  %2 = load i32, ptr @j, align 4
  %3 = icmp eq i32 %2, 0
  %4 = load i8, ptr @a, align 4
  %5 = zext i8 %4 to i32
  %6 = icmp eq i32 %1, 0
  br i1 %6, label %18, label %7

7:                                                ; preds = %0
  %8 = icmp eq i8 %4, 0
  br label %9

9:                                                ; preds = %9, %7
  %10 = phi i32 [ 5, %7 ], [ %16, %9 ]
  %11 = icmp samesign ult i32 %10, 33
  %12 = select i1 %11, i1 %3, i1 false
  %13 = add nuw nsw i32 %10, 1
  %14 = xor i1 %8, true
  %15 = select i1 %12, i1 true, i1 %14
  %16 = select i1 %12, i32 %13, i32 5
  br i1 %15, label %9, label %17, !llvm.loop !10

17:                                               ; preds = %9
  store i32 %10, ptr @g, align 1
  store i32 %5, ptr @f, align 1
  br label %18

18:                                               ; preds = %17, %0
  store i32 0, ptr @e, align 4, !tbaa !6
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0)
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
