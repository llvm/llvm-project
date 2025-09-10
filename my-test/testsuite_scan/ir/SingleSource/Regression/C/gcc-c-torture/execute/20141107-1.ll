; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20141107-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20141107-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i1 @f(i32 noundef %0, i1 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 0
  %4 = xor i1 %1, %3
  ret i1 %4
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @checkf(i32 noundef %0, i1 noundef %1) local_unnamed_addr #1 {
  %3 = tail call i1 @f(i32 noundef %0, i1 noundef %1)
  %4 = icmp eq i32 %0, 0
  %5 = xor i1 %4, %3
  %6 = xor i1 %1, %5
  br i1 %6, label %7, label %8

7:                                                ; preds = %2
  tail call void @abort() #3
  unreachable

8:                                                ; preds = %2
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = tail call i1 @f(i32 noundef 0, i1 noundef false)
  br i1 %1, label %3, label %2

2:                                                ; preds = %0
  tail call void @abort() #3
  unreachable

3:                                                ; preds = %0
  %4 = tail call i1 @f(i32 noundef 0, i1 noundef true)
  br i1 %4, label %5, label %6

5:                                                ; preds = %3
  tail call void @abort() #3
  unreachable

6:                                                ; preds = %3
  %7 = tail call i1 @f(i32 noundef 1, i1 noundef true)
  br i1 %7, label %9, label %8

8:                                                ; preds = %6
  tail call void @abort() #3
  unreachable

9:                                                ; preds = %6
  %10 = tail call i1 @f(i32 noundef 1, i1 noundef false)
  br i1 %10, label %11, label %12

11:                                               ; preds = %9
  tail call void @abort() #3
  unreachable

12:                                               ; preds = %9
  ret i32 0
}

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
