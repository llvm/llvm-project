; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/global.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/global.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@array = dso_local local_unnamed_addr global [4 x i32] [i32 127, i32 -1, i32 100, i32 -28], align 4
@array2 = dso_local local_unnamed_addr global [4 x [4 x i32]] zeroinitializer, align 4
@.str = private unnamed_addr constant [32 x i8] c"error: i=%d, j=%d, result = %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [18 x i8] c"a=%d b=%d a*a=%d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test() local_unnamed_addr #0 {
  %1 = load i32, ptr @array, align 4, !tbaa !6
  %2 = mul nsw i32 %1, %1
  store i32 %2, ptr @array2, align 4, !tbaa !6
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %4, label %7

4:                                                ; preds = %0
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0, i32 noundef 0, i32 noundef %2)
  %6 = load i32, ptr @array, align 4, !tbaa !6
  br label %7

7:                                                ; preds = %0, %4
  %8 = phi i32 [ %1, %0 ], [ %6, %4 ]
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 4), align 4, !tbaa !6
  %10 = mul nsw i32 %9, %8
  store i32 %10, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 4), align 4, !tbaa !6
  %11 = icmp slt i32 %10, 1
  br i1 %11, label %12, label %15

12:                                               ; preds = %7
  %13 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0, i32 noundef 1, i32 noundef %10)
  %14 = load i32, ptr @array, align 4, !tbaa !6
  br label %15

15:                                               ; preds = %12, %7
  %16 = phi i32 [ %14, %12 ], [ %8, %7 ]
  %17 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 8), align 4, !tbaa !6
  %18 = mul nsw i32 %17, %16
  store i32 %18, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 8), align 4, !tbaa !6
  %19 = icmp slt i32 %18, 1
  br i1 %19, label %20, label %23

20:                                               ; preds = %15
  %21 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0, i32 noundef 2, i32 noundef %18)
  %22 = load i32, ptr @array, align 4, !tbaa !6
  br label %23

23:                                               ; preds = %20, %15
  %24 = phi i32 [ %22, %20 ], [ %16, %15 ]
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 12), align 4, !tbaa !6
  %26 = mul nsw i32 %25, %24
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 12), align 4, !tbaa !6
  %27 = icmp slt i32 %26, 1
  br i1 %27, label %28, label %31

28:                                               ; preds = %23
  %29 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 0, i32 noundef 3, i32 noundef %26)
  %30 = load i32, ptr @array, align 4, !tbaa !6
  br label %31

31:                                               ; preds = %28, %23
  %32 = phi i32 [ %30, %28 ], [ %24, %23 ]
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 4), align 4, !tbaa !6
  %34 = mul nsw i32 %32, %33
  store i32 %34, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 16), align 4, !tbaa !6
  %35 = icmp slt i32 %34, 1
  br i1 %35, label %36, label %39

36:                                               ; preds = %31
  %37 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1, i32 noundef 0, i32 noundef %34)
  %38 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 4), align 4, !tbaa !6
  br label %39

39:                                               ; preds = %36, %31
  %40 = phi i32 [ %38, %36 ], [ %33, %31 ]
  %41 = mul nsw i32 %40, %40
  store i32 %41, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 20), align 4, !tbaa !6
  %42 = icmp eq i32 %40, 0
  br i1 %42, label %43, label %46

43:                                               ; preds = %39
  %44 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1, i32 noundef 1, i32 noundef %41)
  %45 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 4), align 4, !tbaa !6
  br label %46

46:                                               ; preds = %43, %39
  %47 = phi i32 [ %45, %43 ], [ %40, %39 ]
  %48 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 8), align 4, !tbaa !6
  %49 = mul nsw i32 %48, %47
  store i32 %49, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 24), align 4, !tbaa !6
  %50 = icmp slt i32 %49, 1
  br i1 %50, label %51, label %54

51:                                               ; preds = %46
  %52 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1, i32 noundef 2, i32 noundef %49)
  %53 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 4), align 4, !tbaa !6
  br label %54

54:                                               ; preds = %51, %46
  %55 = phi i32 [ %53, %51 ], [ %47, %46 ]
  %56 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 12), align 4, !tbaa !6
  %57 = mul nsw i32 %56, %55
  store i32 %57, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 28), align 4, !tbaa !6
  %58 = icmp slt i32 %57, 1
  br i1 %58, label %59, label %61

59:                                               ; preds = %54
  %60 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 1, i32 noundef 3, i32 noundef %57)
  br label %61

61:                                               ; preds = %59, %54
  %62 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 8), align 4, !tbaa !6
  %63 = load i32, ptr @array, align 4, !tbaa !6
  %64 = mul nsw i32 %63, %62
  store i32 %64, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 32), align 4, !tbaa !6
  %65 = icmp slt i32 %64, 1
  br i1 %65, label %66, label %69

66:                                               ; preds = %61
  %67 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 2, i32 noundef 0, i32 noundef %64)
  %68 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 8), align 4, !tbaa !6
  br label %69

69:                                               ; preds = %66, %61
  %70 = phi i32 [ %68, %66 ], [ %62, %61 ]
  %71 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 4), align 4, !tbaa !6
  %72 = mul nsw i32 %71, %70
  store i32 %72, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 36), align 4, !tbaa !6
  %73 = icmp slt i32 %72, 1
  br i1 %73, label %74, label %77

74:                                               ; preds = %69
  %75 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 2, i32 noundef 1, i32 noundef %72)
  %76 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 8), align 4, !tbaa !6
  br label %77

77:                                               ; preds = %74, %69
  %78 = phi i32 [ %76, %74 ], [ %70, %69 ]
  %79 = mul nsw i32 %78, %78
  store i32 %79, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 40), align 4, !tbaa !6
  %80 = icmp eq i32 %78, 0
  br i1 %80, label %81, label %84

81:                                               ; preds = %77
  %82 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 2, i32 noundef 2, i32 noundef %79)
  %83 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 8), align 4, !tbaa !6
  br label %84

84:                                               ; preds = %81, %77
  %85 = phi i32 [ %83, %81 ], [ %78, %77 ]
  %86 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 12), align 4, !tbaa !6
  %87 = mul nsw i32 %86, %85
  store i32 %87, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 44), align 4, !tbaa !6
  %88 = icmp slt i32 %87, 1
  br i1 %88, label %89, label %92

89:                                               ; preds = %84
  %90 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 2, i32 noundef 3, i32 noundef %87)
  %91 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 12), align 4, !tbaa !6
  br label %92

92:                                               ; preds = %89, %84
  %93 = phi i32 [ %91, %89 ], [ %86, %84 ]
  %94 = load i32, ptr @array, align 4, !tbaa !6
  %95 = mul nsw i32 %94, %93
  store i32 %95, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 48), align 4, !tbaa !6
  %96 = icmp slt i32 %95, 1
  br i1 %96, label %97, label %100

97:                                               ; preds = %92
  %98 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 3, i32 noundef 0, i32 noundef %95)
  %99 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 12), align 4, !tbaa !6
  br label %100

100:                                              ; preds = %97, %92
  %101 = phi i32 [ %99, %97 ], [ %93, %92 ]
  %102 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 4), align 4, !tbaa !6
  %103 = mul nsw i32 %102, %101
  store i32 %103, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 52), align 4, !tbaa !6
  %104 = icmp slt i32 %103, 1
  br i1 %104, label %105, label %108

105:                                              ; preds = %100
  %106 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 3, i32 noundef 1, i32 noundef %103)
  %107 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 12), align 4, !tbaa !6
  br label %108

108:                                              ; preds = %105, %100
  %109 = phi i32 [ %107, %105 ], [ %101, %100 ]
  %110 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 8), align 4, !tbaa !6
  %111 = mul nsw i32 %110, %109
  store i32 %111, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 56), align 4, !tbaa !6
  %112 = icmp slt i32 %111, 1
  br i1 %112, label %113, label %116

113:                                              ; preds = %108
  %114 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 3, i32 noundef 2, i32 noundef %111)
  %115 = load i32, ptr getelementptr inbounds nuw (i8, ptr @array, i64 12), align 4, !tbaa !6
  br label %116

116:                                              ; preds = %113, %108
  %117 = phi i32 [ %115, %113 ], [ %109, %108 ]
  %118 = mul nsw i32 %117, %117
  store i32 %118, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 60), align 4, !tbaa !6
  %119 = icmp eq i32 %117, 0
  br i1 %119, label %120, label %122

120:                                              ; preds = %116
  %121 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 3, i32 noundef 3, i32 noundef %118)
  br label %122

122:                                              ; preds = %120, %116
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 127, i32 noundef 100, i32 noundef 16129)
  tail call void @test()
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

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
