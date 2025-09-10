; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr70586.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr70586.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i16 0, align 2
@c = dso_local local_unnamed_addr global i16 0, align 4
@d = dso_local local_unnamed_addr global i16 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @foo(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %10, label %4

4:                                                ; preds = %2
  %5 = icmp ne i32 %0, 0
  %6 = icmp eq i32 %1, 1
  %7 = and i1 %5, %6
  br i1 %7, label %10, label %8

8:                                                ; preds = %4
  %9 = srem i32 %0, %1
  br label %10

10:                                               ; preds = %2, %4, %8
  %11 = phi i32 [ %9, %8 ], [ %0, %4 ], [ %0, %2 ]
  ret i32 %11
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i16, ptr @c, align 4, !tbaa !6
  %2 = sext i16 %1 to i32
  %3 = load i32, ptr @f, align 4, !tbaa !10
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %15, label %5

5:                                                ; preds = %0
  %6 = icmp ne i16 %1, 0
  %7 = icmp eq i32 %3, 1
  %8 = and i1 %6, %7
  br i1 %8, label %9, label %13

9:                                                ; preds = %5
  %10 = load i16, ptr @d, align 4, !tbaa !6
  %11 = srem i16 %10, 2
  %12 = sext i16 %11 to i32
  store i32 %12, ptr @f, align 4, !tbaa !10
  br label %21

13:                                               ; preds = %5
  %14 = srem i32 %2, %3
  br label %15

15:                                               ; preds = %13, %0
  %16 = phi i32 [ %14, %13 ], [ %2, %0 ]
  %17 = load i16, ptr @d, align 4, !tbaa !6
  %18 = srem i16 %17, 2
  %19 = sext i16 %18 to i32
  store i32 %19, ptr @f, align 4, !tbaa !10
  %20 = icmp eq i16 %1, 0
  br i1 %20, label %28, label %21

21:                                               ; preds = %15, %9
  %22 = phi i32 [ %2, %9 ], [ %16, %15 ]
  %23 = icmp eq i16 %1, 1
  %24 = icmp ne i32 %22, 0
  %25 = and i1 %23, %24
  br i1 %25, label %28, label %26

26:                                               ; preds = %21
  %27 = srem i32 %22, %2
  br label %28

28:                                               ; preds = %15, %21, %26
  %29 = phi i32 [ %27, %26 ], [ %22, %21 ], [ %16, %15 ]
  %30 = icmp sgt i32 %29, 5
  %31 = zext i1 %30 to i16
  store i16 %31, ptr @c, align 4, !tbaa !6
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
