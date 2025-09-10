; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr48809.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr48809.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 0, 112) i32 @foo(i8 noundef %0) local_unnamed_addr #0 {
  switch i8 %0, label %21 [
    i8 0, label %2
    i8 1, label %3
    i8 2, label %4
    i8 3, label %5
    i8 4, label %6
    i8 5, label %7
    i8 6, label %8
    i8 7, label %9
    i8 8, label %10
    i8 9, label %11
    i8 10, label %8
    i8 11, label %12
    i8 12, label %13
    i8 13, label %14
    i8 14, label %7
    i8 15, label %15
    i8 16, label %8
    i8 17, label %3
    i8 18, label %4
    i8 19, label %5
    i8 20, label %6
    i8 21, label %16
    i8 22, label %8
    i8 23, label %9
    i8 24, label %10
    i8 25, label %17
    i8 26, label %8
    i8 27, label %18
    i8 28, label %19
    i8 29, label %14
    i8 30, label %7
    i8 31, label %15
    i8 32, label %8
    i8 98, label %20
    i8 -62, label %5
  ]

2:                                                ; preds = %1
  br label %21

3:                                                ; preds = %1, %1
  br label %21

4:                                                ; preds = %1, %1
  br label %21

5:                                                ; preds = %1, %1, %1
  br label %21

6:                                                ; preds = %1, %1
  br label %21

7:                                                ; preds = %1, %1, %1
  br label %21

8:                                                ; preds = %1, %1, %1, %1, %1, %1
  br label %21

9:                                                ; preds = %1, %1
  br label %21

10:                                               ; preds = %1, %1
  br label %21

11:                                               ; preds = %1
  br label %21

12:                                               ; preds = %1
  br label %21

13:                                               ; preds = %1
  br label %21

14:                                               ; preds = %1, %1
  br label %21

15:                                               ; preds = %1, %1
  br label %21

16:                                               ; preds = %1
  br label %21

17:                                               ; preds = %1
  br label %21

18:                                               ; preds = %1
  br label %21

19:                                               ; preds = %1
  br label %21

20:                                               ; preds = %1
  br label %21

21:                                               ; preds = %1, %20, %19, %18, %17, %16, %15, %14, %13, %12, %11, %10, %9, %8, %7, %6, %5, %4, %3, %2
  %22 = phi i32 [ 0, %1 ], [ 1, %2 ], [ 7, %3 ], [ 2, %4 ], [ 19, %5 ], [ 5, %6 ], [ 17, %7 ], [ 31, %8 ], [ 8, %9 ], [ 28, %10 ], [ 16, %11 ], [ 12, %12 ], [ 15, %13 ], [ 111, %14 ], [ 10, %15 ], [ 107, %16 ], [ 106, %17 ], [ 102, %18 ], [ 105, %19 ], [ 18, %20 ]
  ret i32 %22
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
