; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr78720.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr78720.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i64 0, 524289) i64 @foo(i8 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i8 %0, 0
  %3 = select i1 %2, i64 524288, i64 0
  ret i64 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i64 0, 129) i64 @bar(i8 noundef %0) local_unnamed_addr #0 {
  %2 = and i8 %0, -128
  %3 = zext i8 %2 to i64
  ret i64 %3
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i64 0, 33) i64 @baz(i8 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i8 %0, 0
  %3 = select i1 %2, i64 32, i64 0
  ret i64 %3
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call i64 @foo(i8 noundef -1)
  %2 = icmp eq i64 %1, 524288
  br i1 %2, label %3, label %27

3:                                                ; preds = %0
  %4 = tail call i64 @bar(i8 noundef -1)
  %5 = icmp eq i64 %4, 128
  br i1 %5, label %6, label %27

6:                                                ; preds = %3
  %7 = tail call i64 @baz(i8 noundef -1)
  %8 = icmp eq i64 %7, 32
  br i1 %8, label %9, label %27

9:                                                ; preds = %6
  %10 = tail call i64 @foo(i8 noundef 0)
  %11 = icmp eq i64 %10, 0
  br i1 %11, label %12, label %27

12:                                               ; preds = %9
  %13 = tail call i64 @bar(i8 noundef 0)
  %14 = icmp eq i64 %13, 0
  br i1 %14, label %15, label %27

15:                                               ; preds = %12
  %16 = tail call i64 @baz(i8 noundef 0)
  %17 = icmp eq i64 %16, 0
  br i1 %17, label %18, label %27

18:                                               ; preds = %15
  %19 = tail call i64 @foo(i8 noundef 31)
  %20 = icmp eq i64 %19, 0
  br i1 %20, label %21, label %27

21:                                               ; preds = %18
  %22 = tail call i64 @bar(i8 noundef 31)
  %23 = icmp eq i64 %22, 0
  br i1 %23, label %24, label %27

24:                                               ; preds = %21
  %25 = tail call i64 @baz(i8 noundef 31)
  %26 = icmp eq i64 %25, 0
  br i1 %26, label %28, label %27

27:                                               ; preds = %24, %21, %18, %15, %12, %9, %6, %3, %0
  tail call void @abort() #3
  unreachable

28:                                               ; preds = %24
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
