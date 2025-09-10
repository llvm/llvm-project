; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr85529-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr85529-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { i32 }

@c = dso_local local_unnamed_addr global i32 1, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@s = dso_local global %struct.S zeroinitializer, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i8 @foo(i8 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = zext i8 %0 to i32
  %4 = icmp slt i8 %0, 0
  %5 = select i1 %4, i32 0, i32 %1
  %6 = shl i32 %3, %5
  %7 = trunc i32 %6 to i8
  ret i8 %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @d, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %10, label %3

3:                                                ; preds = %0
  %4 = load i32, ptr @e, align 4, !tbaa !6
  %5 = icmp ne i32 %4, 0
  %6 = load i32, ptr @f, align 4
  %7 = icmp ne i32 %6, 0
  %8 = select i1 %5, i1 true, i1 %7
  %9 = zext i1 %8 to i8
  br label %10

10:                                               ; preds = %0, %3
  %11 = phi i8 [ %9, %3 ], [ -83, %0 ]
  %12 = load i32, ptr @b, align 4, !tbaa !6
  %13 = icmp slt i32 %12, 1
  br i1 %13, label %14, label %28

14:                                               ; preds = %10, %23
  %15 = phi i8 [ %24, %23 ], [ %11, %10 ]
  %16 = phi i32 [ %25, %23 ], [ %12, %10 ]
  %17 = load volatile i32, ptr @s, align 4, !tbaa !10
  %18 = icmp slt i8 %15, 0
  %19 = select i1 %18, i8 0, i8 2
  %20 = shl i8 %15, %19
  %21 = icmp slt i8 %15, %20
  br i1 %21, label %22, label %23

22:                                               ; preds = %14
  store i32 0, ptr @c, align 4, !tbaa !6
  br label %23

23:                                               ; preds = %22, %14
  %24 = phi i8 [ 0, %22 ], [ %15, %14 ]
  %25 = add nsw i32 %16, 1
  %26 = icmp eq i32 %16, 0
  br i1 %26, label %27, label %14, !llvm.loop !12

27:                                               ; preds = %23
  store i32 1, ptr @b, align 4, !tbaa !6
  br label %28

28:                                               ; preds = %27, %10
  %29 = load i32, ptr @c, align 4, !tbaa !6
  %30 = icmp eq i32 %29, 1
  br i1 %30, label %32, label %31

31:                                               ; preds = %28
  tail call void @abort() #3
  unreachable

32:                                               ; preds = %28
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !7, i64 0}
!11 = !{!"S", !7, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
