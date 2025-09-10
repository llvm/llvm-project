; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr19005.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr19005.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@v = dso_local local_unnamed_addr global i32 0, align 4
@s = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = load i32, ptr @v, align 4, !tbaa !6
  %4 = load i32, ptr @s, align 4, !tbaa !6
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %14

6:                                                ; preds = %2
  %7 = and i32 %3, 255
  %8 = icmp eq i32 %0, %7
  br i1 %8, label %9, label %13

9:                                                ; preds = %6
  %10 = add i32 %3, 1
  %11 = and i32 %10, 255
  %12 = icmp eq i32 %1, %11
  br i1 %12, label %22, label %13

13:                                               ; preds = %9, %6
  tail call void @abort() #2
  unreachable

14:                                               ; preds = %2
  %15 = add i32 %3, 1
  %16 = and i32 %15, 255
  %17 = icmp eq i32 %0, %16
  %18 = and i32 %3, 255
  %19 = icmp eq i32 %1, %18
  %20 = select i1 %17, i1 %19, i1 false
  br i1 %20, label %22, label %21

21:                                               ; preds = %14
  tail call void @abort() #2
  unreachable

22:                                               ; preds = %14, %9
  %23 = xor i32 %4, 1
  store i32 %23, ptr @s, align 4, !tbaa !6
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add i32 %0, 1
  %3 = and i32 %0, 255
  %4 = and i32 %2, 255
  %5 = load i32, ptr @v, align 4, !tbaa !6
  %6 = load i32, ptr @s, align 4, !tbaa !6
  %7 = icmp eq i32 %6, 0
  br i1 %7, label %8, label %17

8:                                                ; preds = %1
  %9 = and i32 %5, 255
  %10 = icmp eq i32 %3, %9
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  %12 = add i32 %5, 1
  %13 = and i32 %12, 255
  %14 = icmp eq i32 %4, %13
  br i1 %14, label %15, label %16

15:                                               ; preds = %11
  store i32 1, ptr @s, align 4, !tbaa !6
  br label %28

16:                                               ; preds = %11, %8
  tail call void @abort() #2
  unreachable

17:                                               ; preds = %1
  %18 = add i32 %5, 1
  %19 = and i32 %18, 255
  %20 = icmp eq i32 %3, %19
  %21 = and i32 %5, 255
  %22 = icmp eq i32 %4, %21
  %23 = select i1 %20, i1 %22, i1 false
  br i1 %23, label %25, label %24

24:                                               ; preds = %17
  tail call void @abort() #2
  unreachable

25:                                               ; preds = %17
  %26 = xor i32 %6, 1
  store i32 %26, ptr @s, align 4, !tbaa !6
  %27 = icmp eq i32 %6, 1
  br i1 %27, label %35, label %28

28:                                               ; preds = %15, %25
  %29 = phi i32 [ %9, %15 ], [ %21, %25 ]
  %30 = phi i32 [ %13, %15 ], [ %19, %25 ]
  %31 = icmp eq i32 %4, %30
  %32 = icmp eq i32 %3, %29
  %33 = select i1 %31, i1 %32, i1 false
  br i1 %33, label %35, label %34

34:                                               ; preds = %28
  tail call void @abort() #2
  unreachable

35:                                               ; preds = %25, %28
  store i32 %6, ptr @s, align 4, !tbaa !6
  ret i32 0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @s, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br label %3

3:                                                ; preds = %0, %5
  %4 = phi i32 [ -10, %0 ], [ %6, %5 ]
  br i1 %2, label %5, label %8

5:                                                ; preds = %3
  %6 = add nsw i32 %4, 1
  %7 = icmp eq i32 %6, 266
  br i1 %7, label %9, label %3, !llvm.loop !10

8:                                                ; preds = %3
  store i32 %4, ptr @v, align 4, !tbaa !6
  tail call void @abort() #2
  unreachable

9:                                                ; preds = %5
  store i32 266, ptr @v, align 4, !tbaa !6
  ret i32 0
}

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
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
