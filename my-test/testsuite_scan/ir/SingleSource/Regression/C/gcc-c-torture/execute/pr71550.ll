; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr71550.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr71550.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 3, align 4
@h = dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [5 x i8] c"%d%d\00", align 1
@c = dso_local local_unnamed_addr global i32 0, align 4
@f = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global ptr null, align 8
@b = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %55, label %3

3:                                                ; preds = %0
  %4 = load i32, ptr @h, align 4, !tbaa !6
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %6, label %28

6:                                                ; preds = %3
  %7 = load i32, ptr @g, align 4, !tbaa !6
  %8 = freeze i32 %7
  %9 = icmp eq i32 %8, 0
  %10 = load ptr, ptr @e, align 8
  br i1 %9, label %27, label %11

11:                                               ; preds = %6
  %12 = load i32, ptr @d, align 4
  br label %13

13:                                               ; preds = %11, %23
  %14 = phi i32 [ %1, %11 ], [ %25, %23 ]
  %15 = phi i32 [ %12, %11 ], [ %24, %23 ]
  %16 = icmp ult i32 %15, 10
  br i1 %16, label %17, label %23

17:                                               ; preds = %13, %17
  %18 = phi i32 [ %21, %17 ], [ %15, %13 ]
  %19 = load i8, ptr %10, align 1, !tbaa !10
  %20 = zext i8 %19 to i32
  store i32 %20, ptr @b, align 4, !tbaa !6
  %21 = add i32 %18, 1
  store i32 %21, ptr @d, align 4, !tbaa !6
  %22 = icmp eq i32 %21, 10
  br i1 %22, label %23, label %17

23:                                               ; preds = %17, %13
  %24 = phi i32 [ %15, %13 ], [ 10, %17 ]
  %25 = add nsw i32 %14, -1
  store i32 %25, ptr @a, align 4, !tbaa !6
  %26 = icmp eq i32 %25, 0
  br i1 %26, label %55, label %13, !llvm.loop !11

27:                                               ; preds = %6
  store i32 0, ptr @a, align 4, !tbaa !6
  br label %55

28:                                               ; preds = %3, %45
  %29 = phi i32 [ %37, %45 ], [ 1, %3 ]
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %36, label %31

31:                                               ; preds = %28
  %32 = load i32, ptr @c, align 4, !tbaa !6
  %33 = load i32, ptr @f, align 4, !tbaa !6
  %34 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %32, i32 noundef %33) #3
  %35 = load i32, ptr @h, align 4, !tbaa !6
  br label %36

36:                                               ; preds = %31, %28
  %37 = phi i32 [ %35, %31 ], [ 0, %28 ]
  %38 = load i32, ptr @g, align 4, !tbaa !6
  %39 = freeze i32 %38
  %40 = icmp ne i32 %39, 0
  %41 = load i32, ptr @d, align 4
  %42 = load ptr, ptr @e, align 8
  %43 = icmp ult i32 %41, 10
  %44 = select i1 %40, i1 %43, i1 false
  br i1 %44, label %49, label %45

45:                                               ; preds = %49, %36
  %46 = load i32, ptr @a, align 4, !tbaa !6
  %47 = add nsw i32 %46, -1
  store i32 %47, ptr @a, align 4, !tbaa !6
  %48 = icmp eq i32 %47, 0
  br i1 %48, label %55, label %28, !llvm.loop !13

49:                                               ; preds = %36, %49
  %50 = phi i32 [ %53, %49 ], [ %41, %36 ]
  %51 = load i8, ptr %42, align 1, !tbaa !10
  %52 = zext i8 %51 to i32
  store i32 %52, ptr @b, align 4, !tbaa !6
  %53 = add i32 %50, 1
  store i32 %53, ptr @d, align 4, !tbaa !6
  %54 = icmp eq i32 %53, 10
  br i1 %54, label %45, label %49

55:                                               ; preds = %45, %23, %27, %0
  tail call void @exit(i32 noundef 0) #4
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind }

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
!10 = !{!8, !8, i64 0}
!11 = distinct !{!11, !12}
!12 = !{!"llvm.loop.mustprogress"}
!13 = distinct !{!13, !12, !14}
!14 = !{!"llvm.loop.unswitch.partial.disable"}
