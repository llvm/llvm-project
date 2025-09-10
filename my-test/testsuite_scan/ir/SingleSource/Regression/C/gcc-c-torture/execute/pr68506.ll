; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68506.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68506.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@m = dso_local local_unnamed_addr global i32 0, align 4
@n = dso_local local_unnamed_addr global i32 0, align 4
@o = dso_local local_unnamed_addr global i32 0, align 4
@p = dso_local local_unnamed_addr global i32 0, align 4
@s = dso_local local_unnamed_addr global i32 0, align 4
@u = dso_local local_unnamed_addr global i32 0, align 4
@i = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i8 0, align 4
@q = dso_local local_unnamed_addr global i8 0, align 4
@y = dso_local local_unnamed_addr global i8 0, align 4
@d = dso_local local_unnamed_addr global i16 0, align 4
@e = dso_local local_unnamed_addr global i8 0, align 4
@t = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @fn1(i32 noundef returned %0) local_unnamed_addr #0 {
  ret i32 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i8 @fn2(i8 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = icmp sgt i32 %1, 1
  %4 = zext i8 %0 to i32
  %5 = select i1 %3, i32 0, i32 %1
  %6 = lshr i32 %4, %5
  %7 = trunc nuw i32 %6 to i8
  ret i8 %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @p, align 4, !tbaa !6
  %2 = icmp slt i32 %1, 31
  br i1 %2, label %3, label %34

3:                                                ; preds = %0
  %4 = load i8, ptr @c, align 4
  store i8 0, ptr @c, align 4, !tbaa !10
  %5 = load i32, ptr @i, align 4
  %6 = trunc i32 %5 to i8
  %7 = or i8 %6, 7
  %8 = load i32, ptr @b, align 4
  %9 = freeze i32 %8
  %10 = icmp eq i32 %9, 0
  %11 = icmp ult i8 %4, 2
  %12 = zext i1 %11 to i8
  %13 = or i8 %4, %12
  br i1 %10, label %14, label %25

14:                                               ; preds = %3
  %15 = icmp eq i8 %13, 0
  br i1 %15, label %22, label %16

16:                                               ; preds = %14
  %17 = zext i8 %13 to i32
  store i8 0, ptr @q, align 4, !tbaa !10
  %18 = icmp eq i32 %1, 30
  store i32 31, ptr @p, align 4
  %19 = select i1 %18, i32 %17, i32 1
  %20 = icmp eq i32 %19, 30
  %21 = zext i1 %20 to i32
  store i32 %19, ptr @s, align 4, !tbaa !6
  store i32 %19, ptr @t, align 4, !tbaa !6
  store i16 0, ptr @d, align 4, !tbaa !11
  store i32 %19, ptr @m, align 4, !tbaa !6
  store i32 %21, ptr @o, align 4, !tbaa !6
  store i8 %7, ptr @e, align 4, !tbaa !10
  store i8 0, ptr @y, align 4, !tbaa !10
  br label %34

22:                                               ; preds = %14
  %23 = icmp eq i32 %1, 0
  %24 = zext i1 %23 to i32
  br label %32

25:                                               ; preds = %3
  %26 = zext i8 %13 to i32
  %27 = icmp eq i32 %1, %26
  %28 = zext i1 %27 to i32
  %29 = icmp eq i8 %13, 0
  br i1 %29, label %32, label %30

30:                                               ; preds = %25
  store i8 0, ptr @q, align 4, !tbaa !10
  store i32 %26, ptr @s, align 4, !tbaa !6
  store i32 %26, ptr @t, align 4, !tbaa !6
  store i16 0, ptr @d, align 4, !tbaa !11
  store i32 %26, ptr @m, align 4, !tbaa !6
  store i32 %28, ptr @o, align 4, !tbaa !6
  store i8 %7, ptr @e, align 4, !tbaa !10
  store i8 0, ptr @y, align 4, !tbaa !10
  br label %31

31:                                               ; preds = %31, %30
  br label %31

32:                                               ; preds = %25, %22
  %33 = phi i32 [ %24, %22 ], [ %28, %25 ]
  store i32 0, ptr @s, align 4, !tbaa !6
  store i32 0, ptr @t, align 4, !tbaa !6
  store i16 0, ptr @d, align 4, !tbaa !11
  store i32 0, ptr @m, align 4, !tbaa !6
  store i32 %33, ptr @o, align 4, !tbaa !6
  store i8 0, ptr @e, align 4, !tbaa !10
  store i8 0, ptr @y, align 4, !tbaa !10
  tail call void @abort() #3
  unreachable

34:                                               ; preds = %0, %16
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
!10 = !{!8, !8, i64 0}
!11 = !{!12, !12, i64 0}
!12 = !{!"short", !8, i64 0}
