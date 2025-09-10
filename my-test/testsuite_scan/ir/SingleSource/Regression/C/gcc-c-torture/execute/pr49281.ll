; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr49281.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr49281.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 4, 0) i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 2
  %3 = or i32 %2, 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 3, 0) i32 @bar(i32 noundef %0) local_unnamed_addr #0 {
  %2 = shl i32 %0, 2
  %3 = or disjoint i32 %2, 3
  ret i32 %3
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call i32 @foo(i32 noundef 43)
  %2 = icmp eq i32 %1, 172
  br i1 %2, label %3, label %9

3:                                                ; preds = %0
  %4 = tail call i32 @foo(i32 noundef 1)
  %5 = icmp eq i32 %4, 4
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = tail call i32 @foo(i32 noundef 2)
  %8 = icmp eq i32 %7, 12
  br i1 %8, label %10, label %9

9:                                                ; preds = %6, %3, %0
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %6
  %11 = tail call i32 @bar(i32 noundef 43)
  %12 = icmp eq i32 %11, 175
  br i1 %12, label %13, label %19

13:                                               ; preds = %10
  %14 = tail call i32 @bar(i32 noundef 1)
  %15 = icmp eq i32 %14, 7
  br i1 %15, label %16, label %19

16:                                               ; preds = %13
  %17 = tail call i32 @bar(i32 noundef 2)
  %18 = icmp eq i32 %17, 11
  br i1 %18, label %20, label %19

19:                                               ; preds = %16, %13, %10
  tail call void @abort() #3
  unreachable

20:                                               ; preds = %16
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
