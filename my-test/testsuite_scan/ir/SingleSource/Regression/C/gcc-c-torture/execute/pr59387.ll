; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr59387.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr59387.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { i32 }

@d = dso_local global ptr null, align 8
@e = dso_local local_unnamed_addr global ptr @d, align 8
@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global %struct.S zeroinitializer, align 4
@c = dso_local local_unnamed_addr global i8 0, align 4
@f = dso_local global i32 0, align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = load i8, ptr @c, align 4
  %2 = load ptr, ptr @e, align 8, !tbaa !6
  %3 = add i8 %1, 56
  %4 = add i8 %1, -24
  store ptr @f, ptr %2, align 8, !tbaa !12
  %5 = load ptr, ptr @d, align 8, !tbaa !12
  %6 = icmp eq ptr %5, null
  br i1 %6, label %79, label %7

7:                                                ; preds = %0
  %8 = add i8 %1, -48
  %9 = load ptr, ptr @d, align 8, !tbaa !12
  %10 = icmp eq ptr %9, null
  br i1 %10, label %79, label %11

11:                                               ; preds = %7
  %12 = add i8 %1, -72
  %13 = load ptr, ptr @d, align 8, !tbaa !12
  %14 = icmp eq ptr %13, null
  br i1 %14, label %79, label %15

15:                                               ; preds = %11
  %16 = add i8 %1, -96
  %17 = load ptr, ptr @d, align 8, !tbaa !12
  %18 = icmp eq ptr %17, null
  br i1 %18, label %79, label %19

19:                                               ; preds = %15
  %20 = add i8 %1, -120
  %21 = load ptr, ptr @d, align 8, !tbaa !12
  %22 = icmp eq ptr %21, null
  br i1 %22, label %79, label %23

23:                                               ; preds = %19
  %24 = add i8 %1, 112
  %25 = load ptr, ptr @d, align 8, !tbaa !12
  %26 = icmp eq ptr %25, null
  br i1 %26, label %79, label %27

27:                                               ; preds = %23
  %28 = add i8 %1, 88
  %29 = load ptr, ptr @d, align 8, !tbaa !12
  %30 = icmp eq ptr %29, null
  br i1 %30, label %79, label %31

31:                                               ; preds = %27
  %32 = add i8 %1, 64
  %33 = load ptr, ptr @d, align 8, !tbaa !12
  %34 = icmp eq ptr %33, null
  br i1 %34, label %79, label %35

35:                                               ; preds = %31
  %36 = add i8 %1, 40
  %37 = load ptr, ptr @d, align 8, !tbaa !12
  %38 = icmp eq ptr %37, null
  br i1 %38, label %79, label %39

39:                                               ; preds = %35
  %40 = add i8 %1, 16
  %41 = load ptr, ptr @d, align 8, !tbaa !12
  %42 = icmp eq ptr %41, null
  br i1 %42, label %79, label %43

43:                                               ; preds = %39
  %44 = add i8 %1, -8
  %45 = load ptr, ptr @d, align 8, !tbaa !12
  %46 = icmp eq ptr %45, null
  br i1 %46, label %79, label %47

47:                                               ; preds = %43
  %48 = add i8 %1, -32
  %49 = load ptr, ptr @d, align 8, !tbaa !12
  %50 = icmp eq ptr %49, null
  br i1 %50, label %79, label %51

51:                                               ; preds = %47
  %52 = add i8 %1, -56
  %53 = load ptr, ptr @d, align 8, !tbaa !12
  %54 = icmp eq ptr %53, null
  br i1 %54, label %79, label %55

55:                                               ; preds = %51
  %56 = add i8 %1, -80
  %57 = load ptr, ptr @d, align 8, !tbaa !12
  %58 = icmp eq ptr %57, null
  br i1 %58, label %79, label %59

59:                                               ; preds = %55
  %60 = add i8 %1, -104
  %61 = load ptr, ptr @d, align 8, !tbaa !12
  %62 = icmp eq ptr %61, null
  br i1 %62, label %79, label %63

63:                                               ; preds = %59
  %64 = xor i8 %1, -128
  %65 = load ptr, ptr @d, align 8, !tbaa !12
  %66 = icmp eq ptr %65, null
  br i1 %66, label %79, label %67

67:                                               ; preds = %63
  %68 = add i8 %1, 104
  %69 = load ptr, ptr @d, align 8, !tbaa !12
  %70 = icmp eq ptr %69, null
  br i1 %70, label %79, label %71

71:                                               ; preds = %67
  %72 = add i8 %1, 80
  %73 = load ptr, ptr @d, align 8, !tbaa !12
  %74 = icmp eq ptr %73, null
  br i1 %74, label %79, label %75

75:                                               ; preds = %71
  %76 = load ptr, ptr @d, align 8, !tbaa !12
  %77 = icmp eq ptr %76, null
  %78 = sext i1 %77 to i32
  br label %79

79:                                               ; preds = %75, %71, %67, %63, %59, %55, %51, %47, %43, %39, %35, %31, %27, %23, %19, %15, %11, %7, %0
  %80 = phi i8 [ %4, %0 ], [ %8, %7 ], [ %12, %11 ], [ %16, %15 ], [ %20, %19 ], [ %24, %23 ], [ %28, %27 ], [ %32, %31 ], [ %36, %35 ], [ %40, %39 ], [ %44, %43 ], [ %48, %47 ], [ %52, %51 ], [ %56, %55 ], [ %60, %59 ], [ %64, %63 ], [ %68, %67 ], [ %72, %71 ], [ %3, %75 ]
  %81 = phi i32 [ -19, %0 ], [ -18, %7 ], [ -17, %11 ], [ -16, %15 ], [ -15, %19 ], [ -14, %23 ], [ -13, %27 ], [ -12, %31 ], [ -11, %35 ], [ -10, %39 ], [ -9, %43 ], [ -8, %47 ], [ -7, %51 ], [ -6, %55 ], [ -5, %59 ], [ -4, %63 ], [ -3, %67 ], [ -2, %71 ], [ %78, %75 ]
  store i32 24, ptr @b, align 4, !tbaa !14
  store i8 %80, ptr @c, align 4, !tbaa !17
  store i32 %81, ptr @a, align 4, !tbaa !18
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p2 int", !8, i64 0}
!8 = !{!"any p2 pointer", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"p1 int", !9, i64 0}
!14 = !{!15, !16, i64 0}
!15 = !{!"S", !16, i64 0}
!16 = !{!"int", !10, i64 0}
!17 = !{!10, !10, i64 0}
!18 = !{!16, !16, i64 0}
