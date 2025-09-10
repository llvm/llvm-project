; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58431.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58431.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@i = dso_local local_unnamed_addr global i16 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@k = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@j = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local global i32 0, align 4
@a = dso_local local_unnamed_addr global i8 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i8 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i16, ptr @i, align 4, !tbaa !6
  %2 = xor i16 %1, 1
  store i16 %2, ptr @i, align 4, !tbaa !6
  %3 = load i32, ptr @k, align 4, !tbaa !10
  %4 = load i8, ptr @a, align 4, !tbaa !12
  %5 = trunc i16 %2 to i8
  %6 = icmp eq i8 %4, %5
  %7 = load i32, ptr @j, align 4, !tbaa !10
  %8 = icmp eq i32 %7, 0
  br i1 %6, label %9, label %20

9:                                                ; preds = %0
  %10 = load i32, ptr @e, align 4
  br i1 %8, label %11, label %15

11:                                               ; preds = %9
  %12 = load volatile i32, ptr @c, align 4, !tbaa !10
  %13 = icmp ne i32 %12, 0
  %14 = zext i1 %13 to i32
  br label %15

15:                                               ; preds = %11, %9
  %16 = phi i32 [ 1, %9 ], [ %14, %11 ]
  %17 = icmp eq i32 %10, 0
  br i1 %17, label %19, label %18

18:                                               ; preds = %15
  store i32 0, ptr @e, align 4, !tbaa !10
  br label %19

19:                                               ; preds = %18, %15
  store i8 1, ptr @h, align 4, !tbaa !12
  store i32 1, ptr @b, align 4, !tbaa !10
  store i32 %3, ptr @g, align 4, !tbaa !10
  store i32 %16, ptr @j, align 4, !tbaa !10
  br label %33

20:                                               ; preds = %0
  %21 = load i32, ptr @d, align 4
  br i1 %8, label %22, label %26

22:                                               ; preds = %20
  %23 = load volatile i32, ptr @c, align 4, !tbaa !10
  %24 = icmp ne i32 %23, 0
  %25 = zext i1 %24 to i32
  br label %26

26:                                               ; preds = %22, %20
  %27 = phi i32 [ 1, %20 ], [ %25, %22 ]
  %28 = icmp slt i32 %21, 1
  br i1 %28, label %29, label %30

29:                                               ; preds = %26
  store i32 1, ptr @d, align 4, !tbaa !10
  br label %30

30:                                               ; preds = %29, %26
  %31 = load i8, ptr @h, align 4, !tbaa !12
  store i32 1, ptr @b, align 4, !tbaa !10
  store i32 %3, ptr @g, align 4, !tbaa !10
  store i32 %27, ptr @j, align 4, !tbaa !10
  %32 = icmp eq i8 %31, 0
  br i1 %32, label %34, label %33

33:                                               ; preds = %19, %30
  tail call void @abort() #2
  unreachable

34:                                               ; preds = %30
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
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = !{!8, !8, i64 0}
