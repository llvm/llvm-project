; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr46909-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr46909-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef range(i32 -1, 2) i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 13
  %3 = icmp eq i32 %0, 0
  %4 = or i1 %2, %3
  %5 = select i1 %4, i32 -1, i32 1
  ret i32 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  br label %4

1:                                                ; preds = %4
  %2 = add nsw i32 %5, 1
  %3 = icmp eq i32 %2, 30
  br i1 %3, label %14, label %4, !llvm.loop !6

4:                                                ; preds = %0, %1
  %5 = phi i32 [ -10, %0 ], [ %2, %1 ]
  %6 = tail call i32 @foo(i32 noundef %5)
  %7 = icmp eq i32 %5, 0
  %8 = select i1 %7, i32 -1, i32 1
  %9 = icmp eq i32 %5, 13
  %10 = select i1 %9, i32 -2, i32 0
  %11 = add nsw i32 %10, %8
  %12 = icmp eq i32 %6, %11
  br i1 %12, label %1, label %13

13:                                               ; preds = %4
  tail call void @abort() #3
  unreachable

14:                                               ; preds = %1
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
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
