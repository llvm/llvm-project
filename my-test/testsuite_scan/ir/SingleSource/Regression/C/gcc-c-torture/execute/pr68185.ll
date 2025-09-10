; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68185.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68185.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@d = dso_local local_unnamed_addr global i32 1, align 4
@w = dso_local local_unnamed_addr global i32 1, align 4
@o = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i16 0, align 4
@z = dso_local local_unnamed_addr global i32 0, align 4
@u = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@t = dso_local local_unnamed_addr global i16 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@q = dso_local local_unnamed_addr global i16 0, align 4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @d, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %37, label %3

3:                                                ; preds = %0
  %4 = load i32, ptr @o, align 4
  %5 = load i32, ptr @e, align 4
  %6 = icmp eq i32 %5, 0
  %7 = load i32, ptr @b, align 4
  %8 = trunc i32 %7 to i16
  %9 = load i32, ptr @z, align 4
  %10 = load i32, ptr @u, align 4
  %11 = icmp eq i32 %10, 0
  %12 = load i32, ptr @a, align 4
  %13 = icmp eq i32 %12, 0
  %14 = load i16, ptr @t, align 4
  %15 = icmp slt i16 %14, 1
  %16 = load i32, ptr @w, align 4
  %17 = trunc i32 %16 to i8
  %18 = icmp eq i32 %4, 0
  br label %19

19:                                               ; preds = %3, %34
  %20 = phi i8 [ undef, %3 ], [ %30, %34 ]
  %21 = phi i32 [ %1, %3 ], [ %35, %34 ]
  br i1 %18, label %29, label %22

22:                                               ; preds = %19
  br i1 %6, label %23, label %24

23:                                               ; preds = %22, %23
  br label %23

24:                                               ; preds = %22
  store i16 %8, ptr @c, align 4, !tbaa !10
  store i32 %9, ptr @o, align 4, !tbaa !6
  br i1 %11, label %26, label %25

25:                                               ; preds = %24
  br i1 %13, label %27, label %28, !llvm.loop !12

26:                                               ; preds = %24, %26
  br label %26

27:                                               ; preds = %25, %27
  br label %27

28:                                               ; preds = %25, %28
  br label %28

29:                                               ; preds = %19
  %30 = select i1 %15, i8 %17, i8 %20
  %31 = zext i8 %30 to i32
  store i32 %31, ptr @f, align 4, !tbaa !6
  %32 = icmp eq i8 %30, 0
  br i1 %32, label %34, label %33

33:                                               ; preds = %29
  store i16 1, ptr @q, align 4, !tbaa !10
  br label %34

34:                                               ; preds = %33, %29
  %35 = add nsw i32 %21, -1
  store i32 %35, ptr @d, align 4, !tbaa !6
  %36 = icmp eq i32 %35, 0
  br i1 %36, label %37, label %19, !llvm.loop !14

37:                                               ; preds = %34, %0
  %38 = load i16, ptr @q, align 4, !tbaa !10
  %39 = icmp eq i16 %38, 1
  br i1 %39, label %41, label %40

40:                                               ; preds = %37
  tail call void @abort() #2
  unreachable

41:                                               ; preds = %37
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"short", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
