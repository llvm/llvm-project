; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/recursive.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/recursive.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [15 x i8] c"Ack(3,%d): %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [17 x i8] c"Fib(%.1f): %.1f\0A\00", align 1
@.str.2 = private unnamed_addr constant [19 x i8] c"Tak(%d,%d,%d): %d\0A\00", align 1
@.str.3 = private unnamed_addr constant [12 x i8] c"Fib(3): %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [24 x i8] c"Tak(3.0,2.0,1.0): %.1f\0A\00", align 1

; Function Attrs: nofree nosync nounwind memory(none) uwtable
define dso_local range(i32 -2147483647, -2147483648) i32 @ack(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %4, label %7

4:                                                ; preds = %15, %2
  %5 = phi i32 [ %1, %2 ], [ %16, %15 ]
  %6 = add nsw i32 %5, 1
  ret i32 %6

7:                                                ; preds = %2, %15
  %8 = phi i32 [ %16, %15 ], [ %1, %2 ]
  %9 = phi i32 [ %10, %15 ], [ %0, %2 ]
  %10 = add nsw i32 %9, -1
  %11 = icmp eq i32 %8, 0
  br i1 %11, label %15, label %12

12:                                               ; preds = %7
  %13 = add nsw i32 %8, -1
  %14 = tail call i32 @ack(i32 noundef %9, i32 noundef %13)
  br label %15

15:                                               ; preds = %7, %12
  %16 = phi i32 [ %14, %12 ], [ 1, %7 ]
  %17 = icmp eq i32 %10, 0
  br i1 %17, label %4, label %7
}

; Function Attrs: nofree nosync nounwind memory(none) uwtable
define dso_local range(i32 -2147483647, -2147483648) i32 @fib(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp slt i32 %0, 2
  br i1 %2, label %13, label %3

3:                                                ; preds = %1, %3
  %4 = phi i32 [ %8, %3 ], [ %0, %1 ]
  %5 = phi i32 [ %9, %3 ], [ 0, %1 ]
  %6 = add nsw i32 %4, -2
  %7 = tail call i32 @fib(i32 noundef %6)
  %8 = add nsw i32 %4, -1
  %9 = add nsw i32 %7, %5
  %10 = icmp samesign ult i32 %4, 3
  br i1 %10, label %11, label %3

11:                                               ; preds = %3
  %12 = add nsw i32 %9, 1
  br label %13

13:                                               ; preds = %11, %1
  %14 = phi i32 [ 1, %1 ], [ %12, %11 ]
  ret i32 %14
}

; Function Attrs: nofree nosync nounwind memory(none) uwtable
define dso_local double @fibFP(double noundef %0) local_unnamed_addr #0 {
  %2 = fcmp olt double %0, 2.000000e+00
  br i1 %2, label %3, label %5

3:                                                ; preds = %1, %5
  %4 = phi double [ %10, %5 ], [ 1.000000e+00, %1 ]
  ret double %4

5:                                                ; preds = %1
  %6 = fadd double %0, -2.000000e+00
  %7 = tail call double @fibFP(double noundef %6)
  %8 = fadd double %0, -1.000000e+00
  %9 = tail call double @fibFP(double noundef %8)
  %10 = fadd double %7, %9
  br label %3
}

; Function Attrs: nofree nosync nounwind memory(none) uwtable
define dso_local i32 @tak(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = icmp slt i32 %1, %0
  br i1 %4, label %5, label %16

5:                                                ; preds = %3, %5
  %6 = phi i32 [ %14, %5 ], [ %2, %3 ]
  %7 = phi i32 [ %12, %5 ], [ %1, %3 ]
  %8 = phi i32 [ %10, %5 ], [ %0, %3 ]
  %9 = add nsw i32 %8, -1
  %10 = tail call i32 @tak(i32 noundef %9, i32 noundef %7, i32 noundef %6)
  %11 = add nsw i32 %7, -1
  %12 = tail call i32 @tak(i32 noundef %11, i32 noundef %6, i32 noundef %8)
  %13 = add nsw i32 %6, -1
  %14 = tail call i32 @tak(i32 noundef %13, i32 noundef %8, i32 noundef %7)
  %15 = icmp slt i32 %12, %10
  br i1 %15, label %5, label %16

16:                                               ; preds = %5, %3
  %17 = phi i32 [ %2, %3 ], [ %14, %5 ]
  ret i32 %17
}

; Function Attrs: nofree nosync nounwind memory(none) uwtable
define dso_local double @takFP(double noundef %0, double noundef %1, double noundef %2) local_unnamed_addr #0 {
  %4 = fcmp olt double %1, %0
  br i1 %4, label %5, label %16

5:                                                ; preds = %3, %5
  %6 = phi double [ %14, %5 ], [ %2, %3 ]
  %7 = phi double [ %12, %5 ], [ %1, %3 ]
  %8 = phi double [ %10, %5 ], [ %0, %3 ]
  %9 = fadd double %8, -1.000000e+00
  %10 = tail call double @takFP(double noundef %9, double noundef %7, double noundef %6)
  %11 = fadd double %7, -1.000000e+00
  %12 = tail call double @takFP(double noundef %11, double noundef %6, double noundef %8)
  %13 = fadd double %6, -1.000000e+00
  %14 = tail call double @takFP(double noundef %13, double noundef %8, double noundef %7)
  %15 = fcmp olt double %12, %10
  br i1 %15, label %5, label %16

16:                                               ; preds = %5, %3
  %17 = phi double [ %2, %3 ], [ %14, %5 ]
  ret double %17
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #1 {
  %3 = tail call i32 @ack(i32 noundef 3, i32 noundef 11)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 11, i32 noundef %3)
  %5 = tail call double @fibFP(double noundef 3.800000e+01)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, double noundef 3.800000e+01, double noundef %5)
  %7 = tail call i32 @tak(i32 noundef 30, i32 noundef 20, i32 noundef 10)
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef 30, i32 noundef 20, i32 noundef 10, i32 noundef %7)
  %9 = tail call i32 @fib(i32 noundef 3)
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %9)
  %11 = tail call double @takFP(double noundef 3.000000e+00, double noundef 2.000000e+00, double noundef 1.000000e+00)
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, double noundef %11)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

attributes #0 = { nofree nosync nounwind memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
