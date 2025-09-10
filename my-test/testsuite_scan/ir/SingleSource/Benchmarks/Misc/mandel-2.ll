; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/mandel-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/mandel-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@max_i = dso_local local_unnamed_addr global i32 65536, align 4
@.str = private unnamed_addr constant [2 x i8] c"*\00", align 1
@.str.1 = private unnamed_addr constant [2 x i8] c" \00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef double @sqr(double noundef %0) local_unnamed_addr #0 {
  %2 = fmul double %0, %0
  ret double %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef double @cnorm2([2 x double] noundef alignstack(8) %0) local_unnamed_addr #0 {
  %2 = extractvalue [2 x double] %0, 0
  %3 = extractvalue [2 x double] %0, 1
  %4 = fmul double %2, %2
  %5 = fmul double %3, %3
  %6 = fadd double %4, %5
  ret double %6
}

; Function Attrs: nounwind uwtable
define dso_local range(i32 1, 0) i32 @loop([2 x double] noundef alignstack(8) %0) local_unnamed_addr #1 {
  %2 = extractvalue [2 x double] %0, 0
  %3 = extractvalue [2 x double] %0, 1
  %4 = fmul double %2, %2
  %5 = fmul double %3, %3
  %6 = fadd double %4, %5
  %7 = fcmp ugt double %6, 4.000000e+00
  br i1 %7, label %41, label %8

8:                                                ; preds = %1
  %9 = load i32, ptr @max_i, align 4, !tbaa !6
  br label %10

10:                                               ; preds = %8, %31
  %11 = phi i32 [ %32, %31 ], [ %9, %8 ]
  %12 = phi double [ %36, %31 ], [ %3, %8 ]
  %13 = phi double [ %35, %31 ], [ %2, %8 ]
  %14 = phi i32 [ %15, %31 ], [ 1, %8 ]
  %15 = add nuw nsw i32 %14, 1
  %16 = icmp slt i32 %14, %11
  br i1 %16, label %17, label %41

17:                                               ; preds = %10
  %18 = fmul double %13, %13
  %19 = fmul double %12, %12
  %20 = fmul double %13, %12
  %21 = fsub double %18, %19
  %22 = fadd double %20, %20
  %23 = fcmp uno double %21, 0.000000e+00
  br i1 %23, label %24, label %31, !prof !10

24:                                               ; preds = %17
  %25 = fcmp uno double %22, 0.000000e+00
  br i1 %25, label %26, label %31, !prof !10

26:                                               ; preds = %24
  %27 = tail call { double, double } @__muldc3(double noundef %13, double noundef %12, double noundef %13, double noundef %12) #4
  %28 = extractvalue { double, double } %27, 0
  %29 = extractvalue { double, double } %27, 1
  %30 = load i32, ptr @max_i, align 4, !tbaa !6
  br label %31

31:                                               ; preds = %26, %24, %17
  %32 = phi i32 [ %11, %17 ], [ %11, %24 ], [ %30, %26 ]
  %33 = phi double [ %21, %17 ], [ %21, %24 ], [ %28, %26 ]
  %34 = phi double [ %22, %17 ], [ %22, %24 ], [ %29, %26 ]
  %35 = fadd double %2, %33
  %36 = fadd double %3, %34
  %37 = fmul double %35, %35
  %38 = fmul double %36, %36
  %39 = fadd double %37, %38
  %40 = fcmp ugt double %39, 4.000000e+00
  br i1 %40, label %41, label %10, !llvm.loop !11

41:                                               ; preds = %10, %31, %1
  %42 = phi i32 [ 1, %1 ], [ %15, %31 ], [ %15, %10 ]
  ret i32 %42
}

declare { double, double } @__muldc3(double, double, double, double) local_unnamed_addr

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %0, %56
  %2 = phi i32 [ -39, %0 ], [ %58, %56 ]
  %3 = sitofp i32 %2 to double
  %4 = fdiv double %3, 4.000000e+01
  %5 = fadd double %4, -5.000000e-01
  br label %6

6:                                                ; preds = %1, %48
  %7 = phi i32 [ -39, %1 ], [ %54, %48 ]
  %8 = sitofp i32 %7 to double
  %9 = fdiv double %8, 4.000000e+01
  %10 = fmul double %9, 0.000000e+00
  %11 = fadd double %5, %10
  %12 = fmul double %11, %11
  %13 = fmul double %9, %9
  %14 = fadd double %13, %12
  %15 = fcmp ugt double %14, 4.000000e+00
  %16 = load i32, ptr @max_i, align 4, !tbaa !6
  br i1 %15, label %48, label %17

17:                                               ; preds = %6, %38
  %18 = phi i32 [ %39, %38 ], [ %16, %6 ]
  %19 = phi double [ %43, %38 ], [ %9, %6 ]
  %20 = phi double [ %42, %38 ], [ %11, %6 ]
  %21 = phi i32 [ %22, %38 ], [ 1, %6 ]
  %22 = add nuw nsw i32 %21, 1
  %23 = icmp slt i32 %21, %18
  br i1 %23, label %24, label %48

24:                                               ; preds = %17
  %25 = fmul double %20, %20
  %26 = fmul double %19, %19
  %27 = fmul double %19, %20
  %28 = fsub double %25, %26
  %29 = fadd double %27, %27
  %30 = fcmp uno double %28, 0.000000e+00
  br i1 %30, label %31, label %38, !prof !10

31:                                               ; preds = %24
  %32 = fcmp uno double %29, 0.000000e+00
  br i1 %32, label %33, label %38, !prof !10

33:                                               ; preds = %31
  %34 = tail call { double, double } @__muldc3(double noundef %20, double noundef %19, double noundef %20, double noundef %19) #4
  %35 = extractvalue { double, double } %34, 0
  %36 = extractvalue { double, double } %34, 1
  %37 = load i32, ptr @max_i, align 4, !tbaa !6
  br label %38

38:                                               ; preds = %33, %31, %24
  %39 = phi i32 [ %18, %24 ], [ %18, %31 ], [ %37, %33 ]
  %40 = phi double [ %28, %24 ], [ %28, %31 ], [ %35, %33 ]
  %41 = phi double [ %29, %24 ], [ %29, %31 ], [ %36, %33 ]
  %42 = fadd double %11, %40
  %43 = fadd double %9, %41
  %44 = fmul double %42, %42
  %45 = fmul double %43, %43
  %46 = fadd double %44, %45
  %47 = fcmp ugt double %46, 4.000000e+00
  br i1 %47, label %48, label %17, !llvm.loop !11

48:                                               ; preds = %17, %38, %6
  %49 = phi i32 [ %16, %6 ], [ %18, %17 ], [ %39, %38 ]
  %50 = phi i32 [ 1, %6 ], [ %22, %38 ], [ %22, %17 ]
  %51 = icmp sgt i32 %50, %49
  %52 = select i1 %51, ptr @.str, ptr @.str.1
  %53 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) %52)
  %54 = add nsw i32 %7, 1
  %55 = icmp eq i32 %54, 39
  br i1 %55, label %56, label %6, !llvm.loop !13

56:                                               ; preds = %48
  %57 = tail call i32 @putchar(i32 10)
  %58 = add nsw i32 %2, 1
  %59 = icmp eq i32 %58, 39
  br i1 %59, label %60, label %1, !llvm.loop !14

60:                                               ; preds = %56
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #3

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!"branch_weights", i32 1, i32 1048575}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12}
!14 = distinct !{!14, !12}
