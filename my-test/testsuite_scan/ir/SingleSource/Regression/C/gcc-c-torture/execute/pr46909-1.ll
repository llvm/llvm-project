; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr46909-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr46909-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -1, 2) i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 4
  %3 = select i1 %2, i32 -1, i32 1
  ret i32 %3
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call i32 @foo(i32 noundef -10)
  %2 = icmp eq i32 %1, 1
  br i1 %2, label %3, label %61

3:                                                ; preds = %0
  %4 = tail call i32 @foo(i32 noundef -9)
  %5 = icmp eq i32 %4, 1
  br i1 %5, label %6, label %61

6:                                                ; preds = %3
  %7 = tail call i32 @foo(i32 noundef -8)
  %8 = icmp eq i32 %7, 1
  br i1 %8, label %9, label %61

9:                                                ; preds = %6
  %10 = tail call i32 @foo(i32 noundef -7)
  %11 = icmp eq i32 %10, 1
  br i1 %11, label %12, label %61

12:                                               ; preds = %9
  %13 = tail call i32 @foo(i32 noundef -6)
  %14 = icmp eq i32 %13, 1
  br i1 %14, label %15, label %61

15:                                               ; preds = %12
  %16 = tail call i32 @foo(i32 noundef -5)
  %17 = icmp eq i32 %16, 1
  br i1 %17, label %18, label %61

18:                                               ; preds = %15
  %19 = tail call i32 @foo(i32 noundef -4)
  %20 = icmp eq i32 %19, 1
  br i1 %20, label %21, label %61

21:                                               ; preds = %18
  %22 = tail call i32 @foo(i32 noundef -3)
  %23 = icmp eq i32 %22, 1
  br i1 %23, label %24, label %61

24:                                               ; preds = %21
  %25 = tail call i32 @foo(i32 noundef -2)
  %26 = icmp eq i32 %25, 1
  br i1 %26, label %27, label %61

27:                                               ; preds = %24
  %28 = tail call i32 @foo(i32 noundef -1)
  %29 = icmp eq i32 %28, 1
  br i1 %29, label %30, label %61

30:                                               ; preds = %27
  %31 = tail call i32 @foo(i32 noundef 0)
  %32 = icmp eq i32 %31, 1
  br i1 %32, label %33, label %61

33:                                               ; preds = %30
  %34 = tail call i32 @foo(i32 noundef 1)
  %35 = icmp eq i32 %34, 1
  br i1 %35, label %36, label %61

36:                                               ; preds = %33
  %37 = tail call i32 @foo(i32 noundef 2)
  %38 = icmp eq i32 %37, 1
  br i1 %38, label %39, label %61

39:                                               ; preds = %36
  %40 = tail call i32 @foo(i32 noundef 3)
  %41 = icmp eq i32 %40, 1
  br i1 %41, label %42, label %61

42:                                               ; preds = %39
  %43 = tail call i32 @foo(i32 noundef 4)
  %44 = icmp eq i32 %43, -1
  br i1 %44, label %45, label %61

45:                                               ; preds = %42
  %46 = tail call i32 @foo(i32 noundef 5)
  %47 = icmp eq i32 %46, 1
  br i1 %47, label %48, label %61

48:                                               ; preds = %45
  %49 = tail call i32 @foo(i32 noundef 6)
  %50 = icmp eq i32 %49, 1
  br i1 %50, label %51, label %61

51:                                               ; preds = %48
  %52 = tail call i32 @foo(i32 noundef 7)
  %53 = icmp eq i32 %52, 1
  br i1 %53, label %54, label %61

54:                                               ; preds = %51
  %55 = tail call i32 @foo(i32 noundef 8)
  %56 = icmp eq i32 %55, 1
  br i1 %56, label %57, label %61

57:                                               ; preds = %54
  %58 = tail call i32 @foo(i32 noundef 9)
  %59 = icmp eq i32 %58, 1
  br i1 %59, label %60, label %61

60:                                               ; preds = %57
  ret i32 0

61:                                               ; preds = %57, %54, %51, %48, %45, %42, %39, %36, %33, %30, %27, %24, %21, %18, %15, %12, %9, %6, %3, %0
  tail call void @abort() #3
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
