; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr42833.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr42833.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @helper_neon_rshl_s8(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = lshr i32 %0, 8
  %4 = lshr i32 %0, 16
  %5 = shl i32 %1, 24
  %6 = ashr exact i32 %5, 24
  %7 = icmp sgt i32 %6, 7
  br i1 %7, label %27, label %8

8:                                                ; preds = %2
  %9 = icmp slt i32 %6, -8
  br i1 %9, label %10, label %13

10:                                               ; preds = %8
  %11 = shl i32 %0, 24
  %12 = ashr i32 %11, 31
  br label %27

13:                                               ; preds = %8
  %14 = icmp eq i32 %5, -134217728
  br i1 %14, label %27, label %15

15:                                               ; preds = %13
  %16 = icmp slt i32 %6, 0
  br i1 %16, label %17, label %25

17:                                               ; preds = %15
  %18 = shl i32 %0, 24
  %19 = ashr exact i32 %18, 24
  %20 = xor i32 %6, -1
  %21 = shl nuw nsw i32 1, %20
  %22 = add nsw i32 %21, %19
  %23 = sub nsw i32 0, %6
  %24 = ashr i32 %22, %23
  br label %27

25:                                               ; preds = %15
  %26 = shl i32 %0, %6
  br label %27

27:                                               ; preds = %13, %2, %10, %17, %25
  %28 = phi i32 [ %12, %10 ], [ %24, %17 ], [ %26, %25 ], [ 0, %2 ], [ 0, %13 ]
  %29 = shl i32 %1, 16
  %30 = ashr i32 %29, 24
  %31 = icmp sgt i32 %30, 7
  br i1 %31, label %51, label %32

32:                                               ; preds = %27
  %33 = icmp slt i32 %30, -8
  br i1 %33, label %34, label %37

34:                                               ; preds = %32
  %35 = shl i32 %3, 24
  %36 = ashr i32 %35, 31
  br label %51

37:                                               ; preds = %32
  %38 = icmp eq i32 %30, -8
  br i1 %38, label %51, label %39

39:                                               ; preds = %37
  %40 = icmp slt i32 %30, 0
  br i1 %40, label %41, label %49

41:                                               ; preds = %39
  %42 = shl i32 %3, 24
  %43 = ashr exact i32 %42, 24
  %44 = xor i32 %30, -1
  %45 = shl nuw nsw i32 1, %44
  %46 = add nsw i32 %45, %43
  %47 = sub nsw i32 0, %30
  %48 = ashr i32 %46, %47
  br label %51

49:                                               ; preds = %39
  %50 = shl nuw nsw i32 %3, %30
  br label %51

51:                                               ; preds = %37, %27, %34, %41, %49
  %52 = phi i32 [ %36, %34 ], [ %48, %41 ], [ %50, %49 ], [ 0, %27 ], [ 0, %37 ]
  %53 = shl i32 %1, 8
  %54 = ashr i32 %53, 24
  %55 = icmp sgt i32 %54, 7
  br i1 %55, label %75, label %56

56:                                               ; preds = %51
  %57 = icmp slt i32 %54, -8
  br i1 %57, label %58, label %61

58:                                               ; preds = %56
  %59 = shl i32 %4, 24
  %60 = ashr i32 %59, 31
  br label %75

61:                                               ; preds = %56
  %62 = icmp eq i32 %54, -8
  br i1 %62, label %75, label %63

63:                                               ; preds = %61
  %64 = icmp slt i32 %54, 0
  br i1 %64, label %65, label %73

65:                                               ; preds = %63
  %66 = shl i32 %4, 24
  %67 = ashr exact i32 %66, 24
  %68 = xor i32 %54, -1
  %69 = shl nuw nsw i32 1, %68
  %70 = add nsw i32 %69, %67
  %71 = sub nsw i32 0, %54
  %72 = ashr i32 %70, %71
  br label %75

73:                                               ; preds = %63
  %74 = shl nuw nsw i32 %4, %54
  br label %75

75:                                               ; preds = %61, %51, %58, %65, %73
  %76 = phi i32 [ %60, %58 ], [ %72, %65 ], [ %74, %73 ], [ 0, %51 ], [ 0, %61 ]
  %77 = ashr i32 %1, 24
  %78 = icmp sgt i32 %77, 7
  br i1 %78, label %97, label %79

79:                                               ; preds = %75
  %80 = icmp slt i32 %77, -8
  br i1 %80, label %81, label %83

81:                                               ; preds = %79
  %82 = ashr i32 %0, 31
  br label %97

83:                                               ; preds = %79
  %84 = icmp eq i32 %77, -8
  br i1 %84, label %97, label %85

85:                                               ; preds = %83
  %86 = icmp slt i32 %77, 0
  br i1 %86, label %87, label %94

87:                                               ; preds = %85
  %88 = ashr i32 %0, 24
  %89 = xor i32 %77, -1
  %90 = shl nuw nsw i32 1, %89
  %91 = add nsw i32 %90, %88
  %92 = sub nsw i32 0, %77
  %93 = ashr i32 %91, %92
  br label %97

94:                                               ; preds = %85
  %95 = lshr i32 %0, 24
  %96 = shl nuw nsw i32 %95, %77
  br label %97

97:                                               ; preds = %83, %75, %81, %87, %94
  %98 = phi i32 [ %82, %81 ], [ %93, %87 ], [ %96, %94 ], [ 0, %75 ], [ 0, %83 ]
  %99 = shl i32 %98, 24
  %100 = shl i32 %76, 16
  %101 = and i32 %100, 16711680
  %102 = or disjoint i32 %99, %101
  %103 = shl i32 %52, 8
  %104 = and i32 %103, 65280
  %105 = or disjoint i32 %102, %104
  %106 = and i32 %28, 255
  %107 = or disjoint i32 %105, %106
  ret i32 %107
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
