; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr63659.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr63659.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local global i32 0, align 4
@d = dso_local local_unnamed_addr global ptr @b, align 8
@a = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@i = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global i8 0, align 4
@e = dso_local local_unnamed_addr global i8 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  store i32 0, ptr @a, align 4, !tbaa !6
  br label %4

4:                                                ; preds = %3, %0
  %5 = load i32, ptr @c, align 4, !tbaa !6
  %6 = and i32 %5, 255
  %7 = load i32, ptr @h, align 4, !tbaa !6
  %8 = lshr i32 %6, %7
  store i32 %8, ptr @g, align 4, !tbaa !6
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %13, label %10

10:                                               ; preds = %4
  %11 = trunc nuw i32 %8 to i8
  %12 = urem i8 -1, %11
  br label %13

13:                                               ; preds = %4, %10
  %14 = phi i8 [ %12, %10 ], [ -1, %4 ]
  store i8 %14, ptr @f, align 4, !tbaa !10
  %15 = zext i8 %14 to i32
  store i8 %14, ptr @e, align 4, !tbaa !10
  %16 = load ptr, ptr @d, align 8, !tbaa !11
  store i32 %15, ptr %16, align 4, !tbaa !6
  %17 = load i32, ptr @b, align 4, !tbaa !6
  %18 = icmp eq i32 %17, 255
  br i1 %18, label %20, label %19

19:                                               ; preds = %13
  tail call void @abort() #2
  unreachable

20:                                               ; preds = %13
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
!11 = !{!12, !12, i64 0}
!12 = !{!"p1 int", !13, i64 0}
!13 = !{!"any pointer", !8, i64 0}
