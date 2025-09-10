; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/switch-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/switch-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@switch.table.foo = private unnamed_addr constant [8 x i32] [i32 30, i32 31, i32 30, i32 31, i32 31, i32 30, i32 31, i32 30], align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local range(i32 30, 32) i32 @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = add i32 %0, -4
  %3 = icmp ult i32 %2, 8
  br i1 %3, label %4, label %8

4:                                                ; preds = %1
  %5 = zext nneg i32 %2 to i64
  %6 = getelementptr inbounds nuw i32, ptr @switch.table.foo, i64 %5
  %7 = load i32, ptr %6, align 4
  br label %8

8:                                                ; preds = %1, %4
  %9 = phi i32 [ %7, %4 ], [ 31, %1 ]
  ret i32 %9
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %0, %23
  %2 = phi i32 [ -1, %0 ], [ %24, %23 ]
  %3 = icmp ult i32 %2, 12
  %4 = trunc i32 %2 to i12
  %5 = lshr i12 -1456, %4
  %6 = trunc i12 %5 to i1
  %7 = select i1 %3, i1 %6, i1 false
  switch i32 %2, label %16 [
    i32 4, label %8
    i32 6, label %10
    i32 9, label %12
    i32 11, label %14
  ]

8:                                                ; preds = %1
  br i1 %7, label %23, label %9

9:                                                ; preds = %8
  tail call void @abort() #3
  unreachable

10:                                               ; preds = %1
  br i1 %7, label %23, label %11

11:                                               ; preds = %10
  tail call void @abort() #3
  unreachable

12:                                               ; preds = %1
  br i1 %7, label %23, label %13

13:                                               ; preds = %12
  tail call void @abort() #3
  unreachable

14:                                               ; preds = %1
  br i1 %7, label %23, label %15

15:                                               ; preds = %14
  tail call void @abort() #3
  unreachable

16:                                               ; preds = %1
  %17 = trunc i32 %2 to i12
  %18 = lshr i12 1455, %17
  %19 = trunc i12 %18 to i1
  %20 = xor i1 %3, true
  %21 = select i1 %20, i1 true, i1 %19
  br i1 %21, label %23, label %22

22:                                               ; preds = %16
  tail call void @abort() #3
  unreachable

23:                                               ; preds = %8, %12, %16, %14, %10
  %24 = add nsw i32 %2, 1
  %25 = icmp eq i32 %24, 66
  br i1 %25, label %26, label %1, !llvm.loop !6

26:                                               ; preds = %23
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
