; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68249.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr68249.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@k = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global i32 0, align 4
@m = dso_local local_unnamed_addr global i32 0, align 4
@n = dso_local local_unnamed_addr global i32 0, align 4
@l = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@h = dso_local local_unnamed_addr global i8 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn1() local_unnamed_addr #0 {
  %1 = load i32, ptr @k, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  %3 = load i32, ptr @b, align 4, !tbaa !6
  br i1 %2, label %14, label %4

4:                                                ; preds = %0
  %5 = icmp ne i32 %3, 0
  %6 = load i32, ptr @c, align 4
  %7 = icmp ne i32 %6, 0
  %8 = select i1 %5, i1 true, i1 %7
  %9 = zext i1 %8 to i32
  store i32 %9, ptr @m, align 4, !tbaa !6
  %10 = load i32, ptr @n, align 4, !tbaa !6
  %11 = icmp eq i32 %10, 0
  %12 = shl nuw nsw i32 1, %9
  %13 = select i1 %11, i32 %12, i32 1
  store i32 %13, ptr @g, align 4, !tbaa !6
  store i32 0, ptr @k, align 4, !tbaa !6
  br label %14

14:                                               ; preds = %4, %0
  %15 = add nsw i32 %3, 1
  store i32 %15, ptr @l, align 4, !tbaa !6
  %16 = icmp slt i32 %3, 1
  br i1 %16, label %17, label %21

17:                                               ; preds = %14
  %18 = load i32, ptr @a, align 4, !tbaa !6
  %19 = trunc i32 %18 to i8
  %20 = add i8 %19, 1
  store i8 %20, ptr @h, align 4, !tbaa !10
  store i32 1, ptr @b, align 4, !tbaa !6
  br label %21

21:                                               ; preds = %17, %14
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = load i32, ptr @a, align 4, !tbaa !6
  %2 = icmp slt i32 %1, 1
  %3 = load i8, ptr @h, align 4, !tbaa !10
  br i1 %2, label %4, label %78

4:                                                ; preds = %0
  %5 = load i32, ptr @b, align 4
  %6 = load i32, ptr @k, align 4
  %7 = load i32, ptr @c, align 4
  %8 = icmp ne i32 %7, 0
  %9 = load i32, ptr @n, align 4
  %10 = freeze i32 %9
  %11 = icmp eq i32 %10, 0
  %12 = icmp eq i32 %6, 0
  br i1 %11, label %33, label %13

13:                                               ; preds = %4
  br i1 %12, label %18, label %14

14:                                               ; preds = %13
  %15 = icmp ne i32 %5, 0
  %16 = select i1 %15, i1 true, i1 %8
  %17 = zext i1 %16 to i32
  store i32 %17, ptr @m, align 4, !tbaa !6
  store i32 1, ptr @g, align 4, !tbaa !6
  store i32 0, ptr @k, align 4, !tbaa !6
  br label %18

18:                                               ; preds = %14, %13
  %19 = icmp slt i32 %5, 1
  br i1 %19, label %20, label %23

20:                                               ; preds = %18
  %21 = trunc i32 %1 to i8
  %22 = add i8 %21, 1
  store i8 %22, ptr @h, align 4, !tbaa !10
  store i32 1, ptr @b, align 4, !tbaa !6
  br label %23

23:                                               ; preds = %20, %18
  %24 = phi i8 [ %3, %18 ], [ %22, %20 ]
  %25 = phi i32 [ %5, %18 ], [ 1, %20 ]
  %26 = zext i8 %24 to i32
  %27 = icmp slt i32 %7, %26
  br i1 %27, label %28, label %29

28:                                               ; preds = %23
  store i32 0, ptr @g, align 4, !tbaa !6
  br label %29

29:                                               ; preds = %28, %23
  %30 = icmp eq i32 %1, 0
  br i1 %30, label %74, label %31

31:                                               ; preds = %29
  %32 = icmp eq i8 %24, 0
  br label %64

33:                                               ; preds = %4
  br i1 %12, label %39, label %34

34:                                               ; preds = %33
  %35 = icmp ne i32 %5, 0
  %36 = select i1 %35, i1 true, i1 %8
  %37 = zext i1 %36 to i32
  store i32 %37, ptr @m, align 4, !tbaa !6
  %38 = shl nuw nsw i32 1, %37
  store i32 %38, ptr @g, align 4, !tbaa !6
  store i32 0, ptr @k, align 4, !tbaa !6
  br label %39

39:                                               ; preds = %34, %33
  %40 = icmp slt i32 %5, 1
  br i1 %40, label %41, label %44

41:                                               ; preds = %39
  %42 = trunc i32 %1 to i8
  %43 = add i8 %42, 1
  store i8 %43, ptr @h, align 4, !tbaa !10
  store i32 1, ptr @b, align 4, !tbaa !6
  br label %44

44:                                               ; preds = %41, %39
  %45 = phi i8 [ %3, %39 ], [ %43, %41 ]
  %46 = phi i32 [ %5, %39 ], [ 1, %41 ]
  %47 = zext i8 %45 to i32
  %48 = icmp slt i32 %7, %47
  br i1 %48, label %49, label %50

49:                                               ; preds = %44
  store i32 0, ptr @g, align 4, !tbaa !6
  br label %50

50:                                               ; preds = %49, %44
  %51 = icmp eq i32 %1, 0
  br i1 %51, label %74, label %52

52:                                               ; preds = %50
  %53 = icmp eq i8 %45, 0
  br label %54

54:                                               ; preds = %52, %62
  %55 = phi i8 [ %58, %62 ], [ %45, %52 ]
  %56 = phi i32 [ %57, %62 ], [ %1, %52 ]
  %57 = add nsw i32 %56, 1
  %58 = select i1 %53, i8 %55, i8 %45
  %59 = zext i8 %58 to i32
  %60 = icmp slt i32 %7, %59
  br i1 %60, label %61, label %62

61:                                               ; preds = %54
  store i32 0, ptr @g, align 4, !tbaa !6
  br label %62

62:                                               ; preds = %61, %54
  %63 = icmp eq i32 %57, 0
  br i1 %63, label %74, label %54, !llvm.loop !11

64:                                               ; preds = %31, %72
  %65 = phi i8 [ %68, %72 ], [ %24, %31 ]
  %66 = phi i32 [ %67, %72 ], [ %1, %31 ]
  %67 = add nsw i32 %66, 1
  %68 = select i1 %32, i8 %65, i8 %24
  %69 = zext i8 %68 to i32
  %70 = icmp slt i32 %7, %69
  br i1 %70, label %71, label %72

71:                                               ; preds = %64
  store i32 0, ptr @g, align 4, !tbaa !6
  br label %72

72:                                               ; preds = %64, %71
  %73 = icmp eq i32 %67, 0
  br i1 %73, label %74, label %64, !llvm.loop !14

74:                                               ; preds = %29, %72, %50, %62
  %75 = phi i8 [ %45, %62 ], [ %45, %50 ], [ %24, %72 ], [ %24, %29 ]
  %76 = phi i32 [ %5, %50 ], [ %46, %62 ], [ %5, %29 ], [ %25, %72 ]
  %77 = add nsw i32 %76, 1
  store i32 %77, ptr @l, align 4, !tbaa !6
  store i32 1, ptr @a, align 4, !tbaa !6
  br label %78

78:                                               ; preds = %74, %0
  %79 = phi i8 [ %75, %74 ], [ %3, %0 ]
  %80 = icmp eq i8 %79, 1
  br i1 %80, label %82, label %81

81:                                               ; preds = %78
  tail call void @abort() #3
  unreachable

82:                                               ; preds = %78
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!11 = distinct !{!11, !12, !13}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.peeled.count", i32 1}
!14 = distinct !{!14, !12, !13}
