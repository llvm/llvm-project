; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58419.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr58419.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global i16 0, align 4
@p = dso_local local_unnamed_addr global ptr null, align 8
@b = dso_local local_unnamed_addr global i8 0, align 4
@h = dso_local local_unnamed_addr global i16 0, align 4
@e = dso_local local_unnamed_addr global i8 0, align 4
@a = dso_local local_unnamed_addr global i32 0, align 4
@k = dso_local local_unnamed_addr global i32 0, align 4
@i = dso_local local_unnamed_addr global i32 0, align 4
@g = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: noinline nounwind uwtable
define dso_local void @dummy() local_unnamed_addr #0 {
  tail call void asm sideeffect "", ""() #4, !srcloc !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i8 @foo(i32 noundef %0, i32 noundef %1) local_unnamed_addr #1 {
  %3 = sub nsw i32 %0, %1
  %4 = trunc i32 %3 to i8
  ret i8 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local noundef i32 @bar() local_unnamed_addr #2 {
  store i16 1, ptr @c, align 4, !tbaa !7
  %1 = load ptr, ptr @p, align 8, !tbaa !11
  store i32 0, ptr %1, align 4, !tbaa !14
  ret i32 0
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = load i16, ptr @c, align 4
  %2 = load ptr, ptr @p, align 8
  %3 = load i32, ptr @k, align 4, !tbaa !14
  %4 = icmp ne i32 %3, 0
  %5 = icmp ne i16 %1, 0
  %6 = select i1 %4, i1 %5, i1 false
  %7 = load i32, ptr @i, align 4, !tbaa !14
  %8 = icmp slt i32 %7, 1
  %9 = xor i1 %6, %8
  br i1 %9, label %13, label %10

10:                                               ; preds = %0
  store i16 1, ptr @c, align 4, !tbaa !7
  store i32 0, ptr %2, align 4, !tbaa !14
  %11 = load i32, ptr @k, align 4, !tbaa !14
  %12 = load i32, ptr @i, align 4, !tbaa !14
  br label %13

13:                                               ; preds = %10, %0
  %14 = phi i32 [ %12, %10 ], [ %7, %0 ]
  %15 = phi i32 [ %11, %10 ], [ %3, %0 ]
  %16 = phi i16 [ 1, %10 ], [ %1, %0 ]
  %17 = icmp ne i32 %15, 0
  %18 = icmp ne i16 %16, 0
  %19 = select i1 %17, i1 %18, i1 false
  %20 = icmp slt i32 %14, 1
  %21 = xor i1 %19, %20
  br i1 %21, label %25, label %22

22:                                               ; preds = %13
  store i16 1, ptr @c, align 4, !tbaa !7
  store i32 0, ptr %2, align 4, !tbaa !14
  %23 = load i32, ptr @k, align 4, !tbaa !14
  %24 = load i32, ptr @i, align 4, !tbaa !14
  br label %25

25:                                               ; preds = %22, %13
  %26 = phi i32 [ %24, %22 ], [ %14, %13 ]
  %27 = phi i32 [ %23, %22 ], [ %15, %13 ]
  %28 = phi i16 [ 1, %22 ], [ %16, %13 ]
  %29 = icmp ne i32 %27, 0
  %30 = icmp ne i16 %28, 0
  %31 = select i1 %29, i1 %30, i1 false
  %32 = icmp slt i32 %26, 1
  %33 = xor i1 %31, %32
  br i1 %33, label %37, label %34

34:                                               ; preds = %25
  store i16 1, ptr @c, align 4, !tbaa !7
  store i32 0, ptr %2, align 4, !tbaa !14
  %35 = load i32, ptr @k, align 4, !tbaa !14
  %36 = load i32, ptr @i, align 4, !tbaa !14
  br label %37

37:                                               ; preds = %34, %25
  %38 = phi i32 [ %36, %34 ], [ %26, %25 ]
  %39 = phi i32 [ %35, %34 ], [ %27, %25 ]
  %40 = phi i16 [ 1, %34 ], [ %28, %25 ]
  %41 = zext i1 %33 to i32
  store i32 %41, ptr @g, align 4, !tbaa !14
  %42 = icmp ne i32 %39, 0
  %43 = icmp ne i16 %40, 0
  %44 = select i1 %42, i1 %43, i1 false
  %45 = icmp slt i32 %38, 1
  %46 = xor i1 %44, %45
  br i1 %46, label %50, label %47

47:                                               ; preds = %37
  store i16 1, ptr @c, align 4, !tbaa !7
  store i32 0, ptr %2, align 4, !tbaa !14
  %48 = load i32, ptr @k, align 4, !tbaa !14
  %49 = load i32, ptr @i, align 4, !tbaa !14
  br label %50

50:                                               ; preds = %47, %37
  %51 = phi i32 [ %49, %47 ], [ %38, %37 ]
  %52 = phi i32 [ %48, %47 ], [ %39, %37 ]
  %53 = phi i16 [ 1, %47 ], [ %40, %37 ]
  %54 = icmp ne i32 %52, 0
  %55 = icmp ne i16 %53, 0
  %56 = select i1 %54, i1 %55, i1 false
  %57 = icmp slt i32 %51, 1
  %58 = xor i1 %56, %57
  br i1 %58, label %62, label %59

59:                                               ; preds = %50
  store i16 1, ptr @c, align 4, !tbaa !7
  store i32 0, ptr %2, align 4, !tbaa !14
  %60 = load i32, ptr @k, align 4, !tbaa !14
  %61 = load i32, ptr @i, align 4, !tbaa !14
  br label %62

62:                                               ; preds = %59, %50
  %63 = phi i32 [ %61, %59 ], [ %51, %50 ]
  %64 = phi i32 [ %60, %59 ], [ %52, %50 ]
  %65 = phi i16 [ 1, %59 ], [ %53, %50 ]
  %66 = icmp ne i32 %64, 0
  %67 = icmp ne i16 %65, 0
  %68 = select i1 %66, i1 %67, i1 false
  %69 = icmp slt i32 %63, 1
  %70 = xor i1 %68, %69
  br i1 %70, label %74, label %71

71:                                               ; preds = %62
  store i16 1, ptr @c, align 4, !tbaa !7
  store i32 0, ptr %2, align 4, !tbaa !14
  %72 = load i32, ptr @k, align 4, !tbaa !14
  %73 = load i32, ptr @i, align 4, !tbaa !14
  br label %74

74:                                               ; preds = %71, %62
  %75 = phi i32 [ %73, %71 ], [ %63, %62 ]
  %76 = phi i32 [ %72, %71 ], [ %64, %62 ]
  %77 = phi i16 [ 1, %71 ], [ %65, %62 ]
  %78 = icmp ne i32 %76, 0
  %79 = icmp ne i16 %77, 0
  %80 = select i1 %78, i1 %79, i1 false
  %81 = icmp slt i32 %75, 1
  %82 = xor i1 %80, %81
  br i1 %82, label %86, label %83

83:                                               ; preds = %74
  store i16 1, ptr @c, align 4, !tbaa !7
  store i32 0, ptr %2, align 4, !tbaa !14
  %84 = load i32, ptr @k, align 4, !tbaa !14
  %85 = load i32, ptr @i, align 4, !tbaa !14
  br label %86

86:                                               ; preds = %83, %74
  %87 = phi i32 [ %85, %83 ], [ %75, %74 ]
  %88 = phi i32 [ %84, %83 ], [ %76, %74 ]
  %89 = phi i16 [ 1, %83 ], [ %77, %74 ]
  %90 = load i32, ptr @a, align 4, !tbaa !14
  %91 = icmp ne i32 %88, 0
  %92 = icmp ne i16 %89, 0
  %93 = select i1 %91, i1 %92, i1 false
  %94 = icmp slt i32 %87, 1
  %95 = xor i1 %93, %94
  br i1 %95, label %97, label %96

96:                                               ; preds = %86
  store i16 1, ptr @c, align 4, !tbaa !7
  store i32 0, ptr %2, align 4, !tbaa !14
  br label %97

97:                                               ; preds = %96, %86
  %98 = zext i1 %95 to i32
  store i32 %98, ptr @g, align 4, !tbaa !14
  %99 = trunc i32 %90 to i16
  %100 = zext i1 %93 to i8
  store i8 -30, ptr @b, align 4, !tbaa !16
  store i16 %99, ptr @h, align 4, !tbaa !7
  store i8 %100, ptr @e, align 4, !tbaa !16
  tail call void @dummy()
  ret i32 0
}

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 63}
!7 = !{!8, !8, i64 0}
!8 = !{!"short", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"p1 int", !13, i64 0}
!13 = !{!"any pointer", !9, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"int", !9, i64 0}
!16 = !{!9, !9, i64 0}
