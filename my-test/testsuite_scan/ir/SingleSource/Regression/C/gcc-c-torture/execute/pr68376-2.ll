; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68376-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68376-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f1(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 31
  %3 = xor i32 %2, %0
  ret i32 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f2(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, -1
  %3 = sext i1 %2 to i32
  %4 = xor i32 %0, %3
  ret i32 %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f3(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 1
  %3 = sext i1 %2 to i32
  %4 = xor i32 %0, %3
  ret i32 %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f4(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 0
  %3 = sext i1 %2 to i32
  %4 = xor i32 %0, %3
  ret i32 %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f5(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, -1
  %3 = sext i1 %2 to i32
  %4 = xor i32 %0, %3
  ret i32 %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f6(i32 noundef %0) local_unnamed_addr #0 {
  %2 = ashr i32 %0, 31
  %3 = xor i32 %2, %0
  ret i32 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f7(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp sgt i32 %0, 0
  %3 = sext i1 %2 to i32
  %4 = xor i32 %0, %3
  ret i32 %4
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @f8(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 1
  %3 = sext i1 %2 to i32
  %4 = xor i32 %0, %3
  ret i32 %4
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call i32 @f1(i32 noundef 5)
  %2 = icmp eq i32 %1, 5
  br i1 %2, label %3, label %9

3:                                                ; preds = %0
  %4 = tail call i32 @f1(i32 noundef -5)
  %5 = icmp eq i32 %4, 4
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = tail call i32 @f1(i32 noundef 0)
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %6, %3, %0
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %6
  %11 = tail call i32 @f2(i32 noundef 5)
  %12 = icmp eq i32 %11, -6
  br i1 %12, label %13, label %19

13:                                               ; preds = %10
  %14 = tail call i32 @f2(i32 noundef -5)
  %15 = icmp eq i32 %14, -5
  br i1 %15, label %16, label %19

16:                                               ; preds = %13
  %17 = tail call i32 @f2(i32 noundef 0)
  %18 = icmp eq i32 %17, -1
  br i1 %18, label %20, label %19

19:                                               ; preds = %16, %13, %10
  tail call void @abort() #3
  unreachable

20:                                               ; preds = %16
  %21 = tail call i32 @f3(i32 noundef 5)
  %22 = icmp eq i32 %21, 5
  br i1 %22, label %23, label %29

23:                                               ; preds = %20
  %24 = tail call i32 @f3(i32 noundef -5)
  %25 = icmp eq i32 %24, 4
  br i1 %25, label %26, label %29

26:                                               ; preds = %23
  %27 = tail call i32 @f3(i32 noundef 0)
  %28 = icmp eq i32 %27, -1
  br i1 %28, label %30, label %29

29:                                               ; preds = %26, %23, %20
  tail call void @abort() #3
  unreachable

30:                                               ; preds = %26
  %31 = tail call i32 @f4(i32 noundef 5)
  %32 = icmp eq i32 %31, -6
  br i1 %32, label %33, label %39

33:                                               ; preds = %30
  %34 = tail call i32 @f4(i32 noundef -5)
  %35 = icmp eq i32 %34, -5
  br i1 %35, label %36, label %39

36:                                               ; preds = %33
  %37 = tail call i32 @f4(i32 noundef 0)
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %40, label %39

39:                                               ; preds = %36, %33, %30
  tail call void @abort() #3
  unreachable

40:                                               ; preds = %36
  %41 = tail call i32 @f5(i32 noundef 5)
  %42 = icmp eq i32 %41, -6
  br i1 %42, label %43, label %49

43:                                               ; preds = %40
  %44 = tail call i32 @f5(i32 noundef -5)
  %45 = icmp eq i32 %44, -5
  br i1 %45, label %46, label %49

46:                                               ; preds = %43
  %47 = tail call i32 @f5(i32 noundef 0)
  %48 = icmp eq i32 %47, -1
  br i1 %48, label %50, label %49

49:                                               ; preds = %46, %43, %40
  tail call void @abort() #3
  unreachable

50:                                               ; preds = %46
  %51 = tail call i32 @f6(i32 noundef 5)
  %52 = icmp eq i32 %51, 5
  br i1 %52, label %53, label %59

53:                                               ; preds = %50
  %54 = tail call i32 @f6(i32 noundef -5)
  %55 = icmp eq i32 %54, 4
  br i1 %55, label %56, label %59

56:                                               ; preds = %53
  %57 = tail call i32 @f6(i32 noundef 0)
  %58 = icmp eq i32 %57, 0
  br i1 %58, label %60, label %59

59:                                               ; preds = %56, %53, %50
  tail call void @abort() #3
  unreachable

60:                                               ; preds = %56
  %61 = tail call i32 @f7(i32 noundef 5)
  %62 = icmp eq i32 %61, -6
  br i1 %62, label %63, label %69

63:                                               ; preds = %60
  %64 = tail call i32 @f7(i32 noundef -5)
  %65 = icmp eq i32 %64, -5
  br i1 %65, label %66, label %69

66:                                               ; preds = %63
  %67 = tail call i32 @f7(i32 noundef 0)
  %68 = icmp eq i32 %67, 0
  br i1 %68, label %70, label %69

69:                                               ; preds = %66, %63, %60
  tail call void @abort() #3
  unreachable

70:                                               ; preds = %66
  %71 = tail call i32 @f8(i32 noundef 5)
  %72 = icmp eq i32 %71, 5
  br i1 %72, label %73, label %79

73:                                               ; preds = %70
  %74 = tail call i32 @f8(i32 noundef -5)
  %75 = icmp eq i32 %74, 4
  br i1 %75, label %76, label %79

76:                                               ; preds = %73
  %77 = tail call i32 @f8(i32 noundef 0)
  %78 = icmp eq i32 %77, -1
  br i1 %78, label %80, label %79

79:                                               ; preds = %76, %73, %70
  tail call void @abort() #3
  unreachable

80:                                               ; preds = %76
  ret i32 0
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
