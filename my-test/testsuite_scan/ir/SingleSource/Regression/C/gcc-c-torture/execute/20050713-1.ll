; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050713-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20050713-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @foo2([2 x i64] %0, [2 x i64] %1) local_unnamed_addr #0 {
  %3 = extractvalue [2 x i64] %0, 0
  %4 = extractvalue [2 x i64] %0, 1
  %5 = icmp ne i64 %3, 17179869187
  %6 = and i64 %4, 4294967295
  %7 = icmp ne i64 %6, 5
  %8 = select i1 %5, i1 true, i1 %7
  br i1 %8, label %9, label %10

9:                                                ; preds = %2
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %2
  %11 = extractvalue [2 x i64] %1, 1
  %12 = extractvalue [2 x i64] %1, 0
  %13 = icmp ne i64 %12, 30064771078
  %14 = and i64 %11, 4294967295
  %15 = icmp ne i64 %14, 8
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %17, label %18

17:                                               ; preds = %10
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %10
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @foo3([2 x i64] %0, [2 x i64] %1, [2 x i64] %2) local_unnamed_addr #0 {
  %4 = extractvalue [2 x i64] %0, 1
  %5 = extractvalue [2 x i64] %2, 0
  %6 = extractvalue [2 x i64] %2, 1
  %7 = and i64 %4, 4294967295
  %8 = extractvalue [2 x i64] %0, 0
  %9 = icmp ne i64 %8, 17179869187
  %10 = icmp ne i64 %7, 5
  %11 = select i1 %9, i1 true, i1 %10
  br i1 %11, label %12, label %13

12:                                               ; preds = %3
  tail call void @abort() #3
  unreachable

13:                                               ; preds = %3
  %14 = extractvalue [2 x i64] %1, 1
  %15 = and i64 %14, 4294967295
  %16 = extractvalue [2 x i64] %1, 0
  %17 = icmp ne i64 %16, 30064771078
  %18 = icmp ne i64 %15, 8
  %19 = select i1 %17, i1 true, i1 %18
  br i1 %19, label %20, label %21

20:                                               ; preds = %13
  tail call void @abort() #3
  unreachable

21:                                               ; preds = %13
  %22 = icmp ne i64 %5, 42949672969
  %23 = and i64 %6, 4294967295
  %24 = icmp ne i64 %23, 11
  %25 = select i1 %22, i1 true, i1 %24
  br i1 %25, label %26, label %27

26:                                               ; preds = %21
  tail call void @abort() #3
  unreachable

27:                                               ; preds = %21
  ret i32 0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @bar2([2 x i64] %0, [2 x i64] %1) local_unnamed_addr #0 {
  %3 = extractvalue [2 x i64] %1, 1
  %4 = and i64 %3, 4294967295
  %5 = extractvalue [2 x i64] %1, 0
  %6 = icmp ne i64 %5, 17179869187
  %7 = icmp ne i64 %4, 5
  %8 = select i1 %6, i1 true, i1 %7
  br i1 %8, label %9, label %10

9:                                                ; preds = %2
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %2
  %11 = extractvalue [2 x i64] %0, 1
  %12 = and i64 %11, 4294967295
  %13 = extractvalue [2 x i64] %0, 0
  %14 = icmp ne i64 %13, 30064771078
  %15 = icmp ne i64 %12, 8
  %16 = select i1 %14, i1 true, i1 %15
  br i1 %16, label %17, label %18

17:                                               ; preds = %10
  tail call void @abort() #3
  unreachable

18:                                               ; preds = %10
  ret i32 0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @bar3([2 x i64] %0, [2 x i64] %1, [2 x i64] %2) local_unnamed_addr #0 {
  %4 = extractvalue [2 x i64] %1, 1
  %5 = extractvalue [2 x i64] %2, 1
  %6 = and i64 %4, 4294967295
  %7 = and i64 %5, 4294967295
  %8 = extractvalue [2 x i64] %2, 0
  %9 = extractvalue [2 x i64] %1, 0
  %10 = icmp ne i64 %9, 17179869187
  %11 = icmp ne i64 %6, 5
  %12 = select i1 %10, i1 true, i1 %11
  br i1 %12, label %13, label %14

13:                                               ; preds = %3
  tail call void @abort() #3
  unreachable

14:                                               ; preds = %3
  %15 = extractvalue [2 x i64] %0, 1
  %16 = and i64 %15, 4294967295
  %17 = extractvalue [2 x i64] %0, 0
  %18 = icmp ne i64 %17, 30064771078
  %19 = icmp ne i64 %16, 8
  %20 = select i1 %18, i1 true, i1 %19
  br i1 %20, label %21, label %22

21:                                               ; preds = %14
  tail call void @abort() #3
  unreachable

22:                                               ; preds = %14
  %23 = icmp ne i64 %8, 42949672969
  %24 = icmp ne i64 %7, 11
  %25 = select i1 %23, i1 true, i1 %24
  br i1 %25, label %26, label %27

26:                                               ; preds = %22
  tail call void @abort() #3
  unreachable

27:                                               ; preds = %22
  ret i32 0
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @baz3([2 x i64] %0, [2 x i64] %1, [2 x i64] %2) local_unnamed_addr #0 {
  %4 = extractvalue [2 x i64] %0, 1
  %5 = extractvalue [2 x i64] %1, 1
  %6 = and i64 %5, 4294967295
  %7 = and i64 %4, 4294967295
  %8 = extractvalue [2 x i64] %0, 0
  %9 = extractvalue [2 x i64] %1, 0
  %10 = icmp ne i64 %9, 17179869187
  %11 = icmp ne i64 %6, 5
  %12 = select i1 %10, i1 true, i1 %11
  br i1 %12, label %13, label %14

13:                                               ; preds = %3
  tail call void @abort() #3
  unreachable

14:                                               ; preds = %3
  %15 = extractvalue [2 x i64] %2, 1
  %16 = and i64 %15, 4294967295
  %17 = extractvalue [2 x i64] %2, 0
  %18 = icmp ne i64 %17, 30064771078
  %19 = icmp ne i64 %16, 8
  %20 = select i1 %18, i1 true, i1 %19
  br i1 %20, label %21, label %22

21:                                               ; preds = %14
  tail call void @abort() #3
  unreachable

22:                                               ; preds = %14
  %23 = icmp ne i64 %8, 42949672969
  %24 = icmp ne i64 %7, 11
  %25 = select i1 %23, i1 true, i1 %24
  br i1 %25, label %26, label %27

26:                                               ; preds = %22
  tail call void @abort() #3
  unreachable

27:                                               ; preds = %22
  ret i32 0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
