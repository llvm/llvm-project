; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/strlen-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/strlen-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@v0 = dso_local global i32 0, align 4
@v1 = dso_local global i32 1, align 4
@v2 = dso_local global i32 2, align 4
@a = internal constant [2 x [3 x i8]] [[3 x i8] c"1\00\00", [3 x i8] c"12\00"], align 1
@.str = private unnamed_addr constant [26 x i8] c"assertion on line %i: %s\0A\00", align 1
@.str.1 = private unnamed_addr constant [20 x i8] c"strlen (a[v0]) == 1\00", align 1
@.str.2 = private unnamed_addr constant [25 x i8] c"strlen (&a[v0][v0]) == 1\00", align 1
@.str.3 = private unnamed_addr constant [24 x i8] c"strlen (&a[0][v0]) == 1\00", align 1
@.str.4 = private unnamed_addr constant [24 x i8] c"strlen (&a[v0][0]) == 1\00", align 1
@.str.5 = private unnamed_addr constant [20 x i8] c"strlen (a[v1]) == 2\00", align 1
@.str.6 = private unnamed_addr constant [24 x i8] c"strlen (&a[v1][0]) == 2\00", align 1
@.str.7 = private unnamed_addr constant [24 x i8] c"strlen (&a[1][v0]) == 2\00", align 1
@.str.8 = private unnamed_addr constant [25 x i8] c"strlen (&a[v1][v0]) == 2\00", align 1
@.str.9 = private unnamed_addr constant [24 x i8] c"strlen (&a[v1][1]) == 1\00", align 1
@.str.11 = private unnamed_addr constant [25 x i8] c"strlen (&a[v1][v2]) == 0\00", align 1
@.str.12 = private unnamed_addr constant [25 x i8] c"strlen (&a[i0][v0]) == 1\00", align 1
@.str.13 = private unnamed_addr constant [25 x i8] c"strlen (&a[v0][i0]) == 1\00", align 1
@.str.14 = private unnamed_addr constant [25 x i8] c"strlen (&a[v1][i0]) == 2\00", align 1
@.str.15 = private unnamed_addr constant [25 x i8] c"strlen (&a[i1][v0]) == 2\00", align 1
@.str.16 = private unnamed_addr constant [25 x i8] c"strlen (&a[v1][i1]) == 1\00", align 1
@.str.18 = private unnamed_addr constant [24 x i8] c"strlen (a[0] + v0) == 1\00", align 1
@.str.19 = private unnamed_addr constant [24 x i8] c"strlen (a[v0] + 0) == 1\00", align 1
@.str.20 = private unnamed_addr constant [25 x i8] c"strlen (a[v0] + v0) == 1\00", align 1
@.str.21 = private unnamed_addr constant [24 x i8] c"strlen (a[v1] + 0) == 2\00", align 1
@.str.22 = private unnamed_addr constant [24 x i8] c"strlen (a[1] + v0) == 2\00", align 1
@.str.23 = private unnamed_addr constant [25 x i8] c"strlen (a[v1] + v0) == 2\00", align 1
@.str.24 = private unnamed_addr constant [24 x i8] c"strlen (a[v1] + 1) == 1\00", align 1
@.str.25 = private unnamed_addr constant [25 x i8] c"strlen (a[v1] + v1) == 1\00", align 1
@.str.27 = private unnamed_addr constant [25 x i8] c"strlen (a[v1] + v2) == 0\00", align 1
@.str.29 = private unnamed_addr constant [25 x i8] c"strlen (a[i0] + v0) == 1\00", align 1
@.str.30 = private unnamed_addr constant [25 x i8] c"strlen (a[v0] + i0) == 1\00", align 1
@.str.31 = private unnamed_addr constant [25 x i8] c"strlen (a[v1] + i0) == 2\00", align 1
@.str.32 = private unnamed_addr constant [25 x i8] c"strlen (a[i1] + v0) == 2\00", align 1
@.str.33 = private unnamed_addr constant [25 x i8] c"strlen (a[v1] + i1) == 1\00", align 1
@b = internal constant [2 x [2 x [5 x i8]]] [[2 x [5 x i8]] [[5 x i8] c"1\00\00\00\00", [5 x i8] c"12\00\00\00"], [2 x [5 x i8]] [[5 x i8] c"123\00\00", [5 x i8] c"1234\00"]], align 1
@.str.35 = private unnamed_addr constant [23 x i8] c"strlen (b[0][v0]) == 1\00", align 1
@.str.36 = private unnamed_addr constant [23 x i8] c"strlen (b[v0][0]) == 1\00", align 1
@.str.37 = private unnamed_addr constant [27 x i8] c"strlen (&b[0][0][v0]) == 1\00", align 1
@.str.38 = private unnamed_addr constant [27 x i8] c"strlen (&b[0][v0][0]) == 1\00", align 1
@.str.39 = private unnamed_addr constant [27 x i8] c"strlen (&b[v0][0][0]) == 1\00", align 1
@.str.40 = private unnamed_addr constant [28 x i8] c"strlen (&b[0][v0][v0]) == 1\00", align 1
@.str.41 = private unnamed_addr constant [28 x i8] c"strlen (&b[v0][0][v0]) == 1\00", align 1
@.str.42 = private unnamed_addr constant [28 x i8] c"strlen (&b[v0][v0][0]) == 1\00", align 1
@.str.43 = private unnamed_addr constant [23 x i8] c"strlen (b[0][v1]) == 2\00", align 1
@.str.44 = private unnamed_addr constant [23 x i8] c"strlen (b[v1][0]) == 3\00", align 1
@.str.45 = private unnamed_addr constant [27 x i8] c"strlen (&b[0][0][v1]) == 0\00", align 1
@.str.46 = private unnamed_addr constant [27 x i8] c"strlen (&b[0][v1][0]) == 2\00", align 1
@.str.47 = private unnamed_addr constant [28 x i8] c"strlen (&b[0][v1][v1]) == 1\00", align 1
@.str.48 = private unnamed_addr constant [28 x i8] c"strlen (&b[v1][0][v1]) == 2\00", align 1
@.str.49 = private unnamed_addr constant [28 x i8] c"strlen (&b[v1][v1][0]) == 4\00", align 1
@.str.50 = private unnamed_addr constant [28 x i8] c"strlen (&b[v1][v1][1]) == 3\00", align 1
@.str.51 = private unnamed_addr constant [28 x i8] c"strlen (&b[v1][v1][2]) == 2\00", align 1
@.str.52 = private unnamed_addr constant [24 x i8] c"strlen (b[i0][v0]) == 1\00", align 1
@.str.53 = private unnamed_addr constant [24 x i8] c"strlen (b[v0][i0]) == 1\00", align 1
@.str.54 = private unnamed_addr constant [29 x i8] c"strlen (&b[i0][i0][v0]) == 1\00", align 1
@.str.55 = private unnamed_addr constant [29 x i8] c"strlen (&b[i0][v0][i0]) == 1\00", align 1
@.str.56 = private unnamed_addr constant [29 x i8] c"strlen (&b[v0][i0][i0]) == 1\00", align 1
@.str.57 = private unnamed_addr constant [29 x i8] c"strlen (&b[i0][v0][v0]) == 1\00", align 1
@.str.58 = private unnamed_addr constant [29 x i8] c"strlen (&b[v0][i0][v0]) == 1\00", align 1
@.str.59 = private unnamed_addr constant [29 x i8] c"strlen (&b[v0][v0][i0]) == 1\00", align 1
@.str.60 = private unnamed_addr constant [24 x i8] c"strlen (b[i0][v1]) == 2\00", align 1
@.str.61 = private unnamed_addr constant [24 x i8] c"strlen (b[v1][i0]) == 3\00", align 1
@.str.62 = private unnamed_addr constant [29 x i8] c"strlen (&b[i0][i0][v1]) == 0\00", align 1
@.str.63 = private unnamed_addr constant [29 x i8] c"strlen (&b[i0][v1][i0]) == 2\00", align 1
@.str.64 = private unnamed_addr constant [29 x i8] c"strlen (&b[i0][v1][v1]) == 1\00", align 1
@.str.65 = private unnamed_addr constant [29 x i8] c"strlen (&b[v1][i0][v1]) == 2\00", align 1
@.str.66 = private unnamed_addr constant [29 x i8] c"strlen (&b[v1][v1][i0]) == 4\00", align 1
@.str.67 = private unnamed_addr constant [29 x i8] c"strlen (&b[v1][v1][i1]) == 3\00", align 1
@.str.68 = private unnamed_addr constant [29 x i8] c"strlen (&b[v1][v1][i2]) == 2\00", align 1
@.str.69 = private unnamed_addr constant [27 x i8] c"strlen (b[0][0] + v0) == 1\00", align 1
@.str.70 = private unnamed_addr constant [28 x i8] c"strlen (b[0][v0] + v0) == 1\00", align 1
@.str.71 = private unnamed_addr constant [28 x i8] c"strlen (b[v0][0] + v0) == 1\00", align 1
@.str.72 = private unnamed_addr constant [29 x i8] c"strlen (b[v0][v0] + v0) == 1\00", align 1
@.str.73 = private unnamed_addr constant [27 x i8] c"strlen (b[0][0] + v1) == 0\00", align 1
@.str.74 = private unnamed_addr constant [27 x i8] c"strlen (b[0][v1] + 0) == 2\00", align 1
@.str.75 = private unnamed_addr constant [27 x i8] c"strlen (b[v0][0] + 0) == 1\00", align 1
@.str.76 = private unnamed_addr constant [28 x i8] c"strlen (b[v0][v0] + 0) == 1\00", align 1
@.str.77 = private unnamed_addr constant [28 x i8] c"strlen (b[0][v1] + v1) == 1\00", align 1
@.str.78 = private unnamed_addr constant [28 x i8] c"strlen (b[v1][0] + v1) == 2\00", align 1
@.str.79 = private unnamed_addr constant [28 x i8] c"strlen (b[v1][v1] + 0) == 4\00", align 1
@.str.80 = private unnamed_addr constant [28 x i8] c"strlen (b[v1][v1] + 1) == 3\00", align 1
@.str.81 = private unnamed_addr constant [28 x i8] c"strlen (b[v1][v1] + 2) == 2\00", align 1
@.str.82 = private unnamed_addr constant [29 x i8] c"strlen (b[i0][i0] + v0) == 1\00", align 1
@.str.83 = private unnamed_addr constant [29 x i8] c"strlen (b[i0][v0] + v0) == 1\00", align 1
@.str.84 = private unnamed_addr constant [29 x i8] c"strlen (b[v0][i0] + v0) == 1\00", align 1
@.str.85 = private unnamed_addr constant [29 x i8] c"strlen (b[i0][i0] + v1) == 0\00", align 1
@.str.86 = private unnamed_addr constant [29 x i8] c"strlen (b[i0][v1] + i0) == 2\00", align 1
@.str.87 = private unnamed_addr constant [29 x i8] c"strlen (b[v0][i0] + i0) == 1\00", align 1
@.str.88 = private unnamed_addr constant [29 x i8] c"strlen (b[v0][v0] + i0) == 1\00", align 1
@.str.89 = private unnamed_addr constant [29 x i8] c"strlen (b[i0][v1] + v1) == 1\00", align 1
@.str.90 = private unnamed_addr constant [29 x i8] c"strlen (b[v1][i0] + v1) == 2\00", align 1
@.str.91 = private unnamed_addr constant [29 x i8] c"strlen (b[v1][v1] + i0) == 4\00", align 1
@.str.92 = private unnamed_addr constant [29 x i8] c"strlen (b[v1][v1] + i1) == 3\00", align 1
@.str.93 = private unnamed_addr constant [29 x i8] c"strlen (b[v1][v1] + i2) == 2\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_array_ref_2_3() local_unnamed_addr #0 {
  %1 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %2 = sext i32 %1 to i64
  %3 = getelementptr inbounds [3 x i8], ptr @a, i64 %2
  %4 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %3) #4
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %0
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 19, ptr noundef nonnull @.str.1) #4
  tail call void @abort() #5
  unreachable

8:                                                ; preds = %0
  %9 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds [3 x i8], ptr @a, i64 %10
  %12 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds i8, ptr %11, i64 %13
  %15 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %14) #4
  %16 = icmp eq i64 %15, 1
  br i1 %16, label %19, label %17

17:                                               ; preds = %8
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 20, ptr noundef nonnull @.str.2) #4
  tail call void @abort() #5
  unreachable

19:                                               ; preds = %8
  %20 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds i8, ptr @a, i64 %21
  %23 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %22) #4
  %24 = icmp eq i64 %23, 1
  br i1 %24, label %27, label %25

25:                                               ; preds = %19
  %26 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 21, ptr noundef nonnull @.str.3) #4
  tail call void @abort() #5
  unreachable

27:                                               ; preds = %19
  %28 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds [3 x i8], ptr @a, i64 %29
  %31 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %30) #4
  %32 = icmp eq i64 %31, 1
  br i1 %32, label %35, label %33

33:                                               ; preds = %27
  %34 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 22, ptr noundef nonnull @.str.4) #4
  tail call void @abort() #5
  unreachable

35:                                               ; preds = %27
  %36 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds [3 x i8], ptr @a, i64 %37
  %39 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %38) #4
  %40 = icmp eq i64 %39, 2
  br i1 %40, label %43, label %41

41:                                               ; preds = %35
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 24, ptr noundef nonnull @.str.5) #4
  tail call void @abort() #5
  unreachable

43:                                               ; preds = %35
  %44 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds [3 x i8], ptr @a, i64 %45
  %47 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %46) #4
  %48 = icmp eq i64 %47, 2
  br i1 %48, label %51, label %49

49:                                               ; preds = %43
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 25, ptr noundef nonnull @.str.6) #4
  tail call void @abort() #5
  unreachable

51:                                               ; preds = %43
  %52 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 3), i64 %53
  %55 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %54) #4
  %56 = icmp eq i64 %55, 2
  br i1 %56, label %59, label %57

57:                                               ; preds = %51
  %58 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 26, ptr noundef nonnull @.str.7) #4
  tail call void @abort() #5
  unreachable

59:                                               ; preds = %51
  %60 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %61 = sext i32 %60 to i64
  %62 = getelementptr inbounds [3 x i8], ptr @a, i64 %61
  %63 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds i8, ptr %62, i64 %64
  %66 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %65) #4
  %67 = icmp eq i64 %66, 2
  br i1 %67, label %70, label %68

68:                                               ; preds = %59
  %69 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 27, ptr noundef nonnull @.str.8) #4
  tail call void @abort() #5
  unreachable

70:                                               ; preds = %59
  %71 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %72 = sext i32 %71 to i64
  %73 = getelementptr inbounds [3 x i8], ptr @a, i64 %72, i64 1
  %74 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %73) #4
  %75 = icmp eq i64 %74, 1
  br i1 %75, label %78, label %76

76:                                               ; preds = %70
  %77 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 29, ptr noundef nonnull @.str.9) #4
  tail call void @abort() #5
  unreachable

78:                                               ; preds = %70
  %79 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds [3 x i8], ptr @a, i64 %80, i64 1
  %82 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %81) #4
  %83 = icmp eq i64 %82, 1
  br i1 %83, label %86, label %84

84:                                               ; preds = %78
  %85 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 30, ptr noundef nonnull @.str.9) #4
  tail call void @abort() #5
  unreachable

86:                                               ; preds = %78
  %87 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %88 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %89 = sext i32 %88 to i64
  %90 = getelementptr inbounds [3 x i8], ptr @a, i64 %89
  %91 = load volatile i32, ptr @v2, align 4, !tbaa !6
  %92 = sext i32 %91 to i64
  %93 = getelementptr inbounds i8, ptr %90, i64 %92
  %94 = load i8, ptr %93, align 1
  %95 = icmp eq i8 %94, 0
  br i1 %95, label %98, label %96

96:                                               ; preds = %86
  %97 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 33, ptr noundef nonnull @.str.11) #4
  tail call void @abort() #5
  unreachable

98:                                               ; preds = %86
  %99 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %100 = sext i32 %99 to i64
  %101 = getelementptr inbounds [3 x i8], ptr @a, i64 %100
  %102 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %101) #4
  %103 = icmp eq i64 %102, 1
  br i1 %103, label %106, label %104

104:                                              ; preds = %98
  %105 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 39, ptr noundef nonnull @.str.1) #4
  tail call void @abort() #5
  unreachable

106:                                              ; preds = %98
  %107 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %108 = sext i32 %107 to i64
  %109 = getelementptr inbounds [3 x i8], ptr @a, i64 %108
  %110 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %111 = sext i32 %110 to i64
  %112 = getelementptr inbounds i8, ptr %109, i64 %111
  %113 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %112) #4
  %114 = icmp eq i64 %113, 1
  br i1 %114, label %117, label %115

115:                                              ; preds = %106
  %116 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 40, ptr noundef nonnull @.str.2) #4
  tail call void @abort() #5
  unreachable

117:                                              ; preds = %106
  %118 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %119 = sext i32 %118 to i64
  %120 = getelementptr inbounds i8, ptr @a, i64 %119
  %121 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %120) #4
  %122 = icmp eq i64 %121, 1
  br i1 %122, label %125, label %123

123:                                              ; preds = %117
  %124 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 41, ptr noundef nonnull @.str.12) #4
  tail call void @abort() #5
  unreachable

125:                                              ; preds = %117
  %126 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %127 = sext i32 %126 to i64
  %128 = getelementptr inbounds [3 x i8], ptr @a, i64 %127
  %129 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %128) #4
  %130 = icmp eq i64 %129, 1
  br i1 %130, label %133, label %131

131:                                              ; preds = %125
  %132 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 42, ptr noundef nonnull @.str.13) #4
  tail call void @abort() #5
  unreachable

133:                                              ; preds = %125
  %134 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %135 = sext i32 %134 to i64
  %136 = getelementptr inbounds [3 x i8], ptr @a, i64 %135
  %137 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %136) #4
  %138 = icmp eq i64 %137, 2
  br i1 %138, label %141, label %139

139:                                              ; preds = %133
  %140 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 44, ptr noundef nonnull @.str.5) #4
  tail call void @abort() #5
  unreachable

141:                                              ; preds = %133
  %142 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %143 = sext i32 %142 to i64
  %144 = getelementptr inbounds [3 x i8], ptr @a, i64 %143
  %145 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %144) #4
  %146 = icmp eq i64 %145, 2
  br i1 %146, label %149, label %147

147:                                              ; preds = %141
  %148 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 45, ptr noundef nonnull @.str.14) #4
  tail call void @abort() #5
  unreachable

149:                                              ; preds = %141
  %150 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %151 = sext i32 %150 to i64
  %152 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 3), i64 %151
  %153 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %152) #4
  %154 = icmp eq i64 %153, 2
  br i1 %154, label %157, label %155

155:                                              ; preds = %149
  %156 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 46, ptr noundef nonnull @.str.15) #4
  tail call void @abort() #5
  unreachable

157:                                              ; preds = %149
  %158 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %159 = sext i32 %158 to i64
  %160 = getelementptr inbounds [3 x i8], ptr @a, i64 %159
  %161 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %162 = sext i32 %161 to i64
  %163 = getelementptr inbounds i8, ptr %160, i64 %162
  %164 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %163) #4
  %165 = icmp eq i64 %164, 2
  br i1 %165, label %168, label %166

166:                                              ; preds = %157
  %167 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 47, ptr noundef nonnull @.str.8) #4
  tail call void @abort() #5
  unreachable

168:                                              ; preds = %157
  %169 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %170 = sext i32 %169 to i64
  %171 = getelementptr inbounds [3 x i8], ptr @a, i64 %170, i64 1
  %172 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %171) #4
  %173 = icmp eq i64 %172, 1
  br i1 %173, label %176, label %174

174:                                              ; preds = %168
  %175 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 49, ptr noundef nonnull @.str.16) #4
  tail call void @abort() #5
  unreachable

176:                                              ; preds = %168
  %177 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %178 = sext i32 %177 to i64
  %179 = getelementptr inbounds [3 x i8], ptr @a, i64 %178, i64 1
  %180 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %179) #4
  %181 = icmp eq i64 %180, 1
  br i1 %181, label %184, label %182

182:                                              ; preds = %176
  %183 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 50, ptr noundef nonnull @.str.16) #4
  tail call void @abort() #5
  unreachable

184:                                              ; preds = %176
  %185 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %186 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %187 = sext i32 %186 to i64
  %188 = getelementptr inbounds [3 x i8], ptr @a, i64 %187
  %189 = load volatile i32, ptr @v2, align 4, !tbaa !6
  %190 = sext i32 %189 to i64
  %191 = getelementptr inbounds i8, ptr %188, i64 %190
  %192 = load i8, ptr %191, align 1
  %193 = icmp eq i8 %192, 0
  br i1 %193, label %196, label %194

194:                                              ; preds = %184
  %195 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 53, ptr noundef nonnull @.str.11) #4
  tail call void @abort() #5
  unreachable

196:                                              ; preds = %184
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_array_off_2_3() local_unnamed_addr #0 {
  %1 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %2 = sext i32 %1 to i64
  %3 = getelementptr inbounds i8, ptr @a, i64 %2
  %4 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %3) #4
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %0
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 59, ptr noundef nonnull @.str.18) #4
  tail call void @abort() #5
  unreachable

8:                                                ; preds = %0
  %9 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds [3 x i8], ptr @a, i64 %10
  %12 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %11) #4
  %13 = icmp eq i64 %12, 1
  br i1 %13, label %16, label %14

14:                                               ; preds = %8
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 60, ptr noundef nonnull @.str.19) #4
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %8
  %17 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds [3 x i8], ptr @a, i64 %18
  %20 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds i8, ptr %19, i64 %21
  %23 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %22) #4
  %24 = icmp eq i64 %23, 1
  br i1 %24, label %27, label %25

25:                                               ; preds = %16
  %26 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 61, ptr noundef nonnull @.str.20) #4
  tail call void @abort() #5
  unreachable

27:                                               ; preds = %16
  %28 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %29 = sext i32 %28 to i64
  %30 = getelementptr inbounds [3 x i8], ptr @a, i64 %29
  %31 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %30) #4
  %32 = icmp eq i64 %31, 2
  br i1 %32, label %35, label %33

33:                                               ; preds = %27
  %34 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 63, ptr noundef nonnull @.str.21) #4
  tail call void @abort() #5
  unreachable

35:                                               ; preds = %27
  %36 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 3), i64 %37
  %39 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %38) #4
  %40 = icmp eq i64 %39, 2
  br i1 %40, label %43, label %41

41:                                               ; preds = %35
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 64, ptr noundef nonnull @.str.22) #4
  tail call void @abort() #5
  unreachable

43:                                               ; preds = %35
  %44 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds [3 x i8], ptr @a, i64 %45
  %47 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %46) #4
  %48 = icmp eq i64 %47, 2
  br i1 %48, label %51, label %49

49:                                               ; preds = %43
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 65, ptr noundef nonnull @.str.21) #4
  tail call void @abort() #5
  unreachable

51:                                               ; preds = %43
  %52 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds [3 x i8], ptr @a, i64 %53
  %55 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %56 = sext i32 %55 to i64
  %57 = getelementptr inbounds i8, ptr %54, i64 %56
  %58 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %57) #4
  %59 = icmp eq i64 %58, 2
  br i1 %59, label %62, label %60

60:                                               ; preds = %51
  %61 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 66, ptr noundef nonnull @.str.23) #4
  tail call void @abort() #5
  unreachable

62:                                               ; preds = %51
  %63 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds [3 x i8], ptr @a, i64 %64, i64 1
  %66 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %65) #4
  %67 = icmp eq i64 %66, 1
  br i1 %67, label %70, label %68

68:                                               ; preds = %62
  %69 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 68, ptr noundef nonnull @.str.24) #4
  tail call void @abort() #5
  unreachable

70:                                               ; preds = %62
  %71 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %72 = sext i32 %71 to i64
  %73 = getelementptr inbounds [3 x i8], ptr @a, i64 %72
  %74 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds i8, ptr %73, i64 %75
  %77 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %76) #4
  %78 = icmp eq i64 %77, 1
  br i1 %78, label %81, label %79

79:                                               ; preds = %70
  %80 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 69, ptr noundef nonnull @.str.25) #4
  tail call void @abort() #5
  unreachable

81:                                               ; preds = %70
  %82 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %83 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds [3 x i8], ptr @a, i64 %84
  %86 = load volatile i32, ptr @v2, align 4, !tbaa !6
  %87 = sext i32 %86 to i64
  %88 = getelementptr inbounds i8, ptr %85, i64 %87
  %89 = load i8, ptr %88, align 1
  %90 = icmp eq i8 %89, 0
  br i1 %90, label %93, label %91

91:                                               ; preds = %81
  %92 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 72, ptr noundef nonnull @.str.27) #4
  tail call void @abort() #5
  unreachable

93:                                               ; preds = %81
  %94 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %95 = sext i32 %94 to i64
  %96 = getelementptr inbounds i8, ptr @a, i64 %95
  %97 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %96) #4
  %98 = icmp eq i64 %97, 1
  br i1 %98, label %101, label %99

99:                                               ; preds = %93
  %100 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 79, ptr noundef nonnull @.str.29) #4
  tail call void @abort() #5
  unreachable

101:                                              ; preds = %93
  %102 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %103 = sext i32 %102 to i64
  %104 = getelementptr inbounds [3 x i8], ptr @a, i64 %103
  %105 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %104) #4
  %106 = icmp eq i64 %105, 1
  br i1 %106, label %109, label %107

107:                                              ; preds = %101
  %108 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 80, ptr noundef nonnull @.str.30) #4
  tail call void @abort() #5
  unreachable

109:                                              ; preds = %101
  %110 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %111 = sext i32 %110 to i64
  %112 = getelementptr inbounds [3 x i8], ptr @a, i64 %111
  %113 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %114 = sext i32 %113 to i64
  %115 = getelementptr inbounds i8, ptr %112, i64 %114
  %116 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %115) #4
  %117 = icmp eq i64 %116, 1
  br i1 %117, label %120, label %118

118:                                              ; preds = %109
  %119 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 81, ptr noundef nonnull @.str.20) #4
  tail call void @abort() #5
  unreachable

120:                                              ; preds = %109
  %121 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %122 = sext i32 %121 to i64
  %123 = getelementptr inbounds [3 x i8], ptr @a, i64 %122
  %124 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %123) #4
  %125 = icmp eq i64 %124, 2
  br i1 %125, label %128, label %126

126:                                              ; preds = %120
  %127 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 83, ptr noundef nonnull @.str.31) #4
  tail call void @abort() #5
  unreachable

128:                                              ; preds = %120
  %129 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %130 = sext i32 %129 to i64
  %131 = getelementptr inbounds i8, ptr getelementptr inbounds nuw (i8, ptr @a, i64 3), i64 %130
  %132 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %131) #4
  %133 = icmp eq i64 %132, 2
  br i1 %133, label %136, label %134

134:                                              ; preds = %128
  %135 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 84, ptr noundef nonnull @.str.32) #4
  tail call void @abort() #5
  unreachable

136:                                              ; preds = %128
  %137 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %138 = sext i32 %137 to i64
  %139 = getelementptr inbounds [3 x i8], ptr @a, i64 %138
  %140 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %139) #4
  %141 = icmp eq i64 %140, 2
  br i1 %141, label %144, label %142

142:                                              ; preds = %136
  %143 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 85, ptr noundef nonnull @.str.31) #4
  tail call void @abort() #5
  unreachable

144:                                              ; preds = %136
  %145 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %146 = sext i32 %145 to i64
  %147 = getelementptr inbounds [3 x i8], ptr @a, i64 %146
  %148 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %149 = sext i32 %148 to i64
  %150 = getelementptr inbounds i8, ptr %147, i64 %149
  %151 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %150) #4
  %152 = icmp eq i64 %151, 2
  br i1 %152, label %155, label %153

153:                                              ; preds = %144
  %154 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 86, ptr noundef nonnull @.str.23) #4
  tail call void @abort() #5
  unreachable

155:                                              ; preds = %144
  %156 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %157 = sext i32 %156 to i64
  %158 = getelementptr inbounds [3 x i8], ptr @a, i64 %157, i64 1
  %159 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %158) #4
  %160 = icmp eq i64 %159, 1
  br i1 %160, label %163, label %161

161:                                              ; preds = %155
  %162 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 88, ptr noundef nonnull @.str.33) #4
  tail call void @abort() #5
  unreachable

163:                                              ; preds = %155
  %164 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %165 = sext i32 %164 to i64
  %166 = getelementptr inbounds [3 x i8], ptr @a, i64 %165
  %167 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %168 = sext i32 %167 to i64
  %169 = getelementptr inbounds i8, ptr %166, i64 %168
  %170 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %169) #4
  %171 = icmp eq i64 %170, 1
  br i1 %171, label %174, label %172

172:                                              ; preds = %163
  %173 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 89, ptr noundef nonnull @.str.25) #4
  tail call void @abort() #5
  unreachable

174:                                              ; preds = %163
  %175 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %176 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %177 = sext i32 %176 to i64
  %178 = getelementptr inbounds [3 x i8], ptr @a, i64 %177
  %179 = load volatile i32, ptr @v2, align 4, !tbaa !6
  %180 = sext i32 %179 to i64
  %181 = getelementptr inbounds i8, ptr %178, i64 %180
  %182 = load i8, ptr %181, align 1
  %183 = icmp eq i8 %182, 0
  br i1 %183, label %186, label %184

184:                                              ; preds = %174
  %185 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 92, ptr noundef nonnull @.str.27) #4
  tail call void @abort() #5
  unreachable

186:                                              ; preds = %174
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_array_ref_2_2_5() local_unnamed_addr #0 {
  %1 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %2 = sext i32 %1 to i64
  %3 = getelementptr inbounds [5 x i8], ptr @b, i64 %2
  %4 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %3) #4
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %0
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 97, ptr noundef nonnull @.str.35) #4
  tail call void @abort() #5
  unreachable

8:                                                ; preds = %0
  %9 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %10
  %12 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %11) #4
  %13 = icmp eq i64 %12, 1
  br i1 %13, label %16, label %14

14:                                               ; preds = %8
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 98, ptr noundef nonnull @.str.36) #4
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %8
  %17 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %18 = sext i32 %17 to i64
  %19 = getelementptr inbounds i8, ptr @b, i64 %18
  %20 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %19) #4
  %21 = icmp eq i64 %20, 1
  br i1 %21, label %24, label %22

22:                                               ; preds = %16
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 100, ptr noundef nonnull @.str.37) #4
  tail call void @abort() #5
  unreachable

24:                                               ; preds = %16
  %25 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %26 = sext i32 %25 to i64
  %27 = getelementptr inbounds [5 x i8], ptr @b, i64 %26
  %28 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %27) #4
  %29 = icmp eq i64 %28, 1
  br i1 %29, label %32, label %30

30:                                               ; preds = %24
  %31 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 101, ptr noundef nonnull @.str.38) #4
  tail call void @abort() #5
  unreachable

32:                                               ; preds = %24
  %33 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %34 = sext i32 %33 to i64
  %35 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %34
  %36 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %35) #4
  %37 = icmp eq i64 %36, 1
  br i1 %37, label %40, label %38

38:                                               ; preds = %32
  %39 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 102, ptr noundef nonnull @.str.39) #4
  tail call void @abort() #5
  unreachable

40:                                               ; preds = %32
  %41 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %42 = sext i32 %41 to i64
  %43 = getelementptr inbounds [5 x i8], ptr @b, i64 %42
  %44 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %45 = sext i32 %44 to i64
  %46 = getelementptr inbounds i8, ptr %43, i64 %45
  %47 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %46) #4
  %48 = icmp eq i64 %47, 1
  br i1 %48, label %51, label %49

49:                                               ; preds = %40
  %50 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 104, ptr noundef nonnull @.str.40) #4
  tail call void @abort() #5
  unreachable

51:                                               ; preds = %40
  %52 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %53 = sext i32 %52 to i64
  %54 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %53
  %55 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %56 = sext i32 %55 to i64
  %57 = getelementptr inbounds i8, ptr %54, i64 %56
  %58 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %57) #4
  %59 = icmp eq i64 %58, 1
  br i1 %59, label %62, label %60

60:                                               ; preds = %51
  %61 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 105, ptr noundef nonnull @.str.41) #4
  tail call void @abort() #5
  unreachable

62:                                               ; preds = %51
  %63 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %64
  %66 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %67 = sext i32 %66 to i64
  %68 = getelementptr inbounds [5 x i8], ptr %65, i64 %67
  %69 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %68) #4
  %70 = icmp eq i64 %69, 1
  br i1 %70, label %73, label %71

71:                                               ; preds = %62
  %72 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 106, ptr noundef nonnull @.str.42) #4
  tail call void @abort() #5
  unreachable

73:                                               ; preds = %62
  %74 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %75 = sext i32 %74 to i64
  %76 = getelementptr inbounds [5 x i8], ptr @b, i64 %75
  %77 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %76) #4
  %78 = icmp eq i64 %77, 2
  br i1 %78, label %81, label %79

79:                                               ; preds = %73
  %80 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 108, ptr noundef nonnull @.str.43) #4
  tail call void @abort() #5
  unreachable

81:                                               ; preds = %73
  %82 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %83 = sext i32 %82 to i64
  %84 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %83
  %85 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %84) #4
  %86 = icmp eq i64 %85, 3
  br i1 %86, label %89, label %87

87:                                               ; preds = %81
  %88 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 109, ptr noundef nonnull @.str.44) #4
  tail call void @abort() #5
  unreachable

89:                                               ; preds = %81
  %90 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %91 = sext i32 %90 to i64
  %92 = getelementptr inbounds i8, ptr @b, i64 %91
  %93 = load i8, ptr %92, align 1
  %94 = icmp eq i8 %93, 0
  br i1 %94, label %97, label %95

95:                                               ; preds = %89
  %96 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 111, ptr noundef nonnull @.str.45) #4
  tail call void @abort() #5
  unreachable

97:                                               ; preds = %89
  %98 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %99 = sext i32 %98 to i64
  %100 = getelementptr inbounds [5 x i8], ptr @b, i64 %99
  %101 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %100) #4
  %102 = icmp eq i64 %101, 2
  br i1 %102, label %105, label %103

103:                                              ; preds = %97
  %104 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 112, ptr noundef nonnull @.str.46) #4
  tail call void @abort() #5
  unreachable

105:                                              ; preds = %97
  %106 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %107 = sext i32 %106 to i64
  %108 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %107
  %109 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %108) #4
  %110 = icmp eq i64 %109, 1
  br i1 %110, label %113, label %111

111:                                              ; preds = %105
  %112 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 113, ptr noundef nonnull @.str.39) #4
  tail call void @abort() #5
  unreachable

113:                                              ; preds = %105
  %114 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %115 = sext i32 %114 to i64
  %116 = getelementptr inbounds [5 x i8], ptr @b, i64 %115
  %117 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %118 = sext i32 %117 to i64
  %119 = getelementptr inbounds i8, ptr %116, i64 %118
  %120 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %119) #4
  %121 = icmp eq i64 %120, 1
  br i1 %121, label %124, label %122

122:                                              ; preds = %113
  %123 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 115, ptr noundef nonnull @.str.40) #4
  tail call void @abort() #5
  unreachable

124:                                              ; preds = %113
  %125 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %126 = sext i32 %125 to i64
  %127 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %126
  %128 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %129 = sext i32 %128 to i64
  %130 = getelementptr inbounds i8, ptr %127, i64 %129
  %131 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %130) #4
  %132 = icmp eq i64 %131, 1
  br i1 %132, label %135, label %133

133:                                              ; preds = %124
  %134 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 116, ptr noundef nonnull @.str.41) #4
  tail call void @abort() #5
  unreachable

135:                                              ; preds = %124
  %136 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %137 = sext i32 %136 to i64
  %138 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %137
  %139 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %140 = sext i32 %139 to i64
  %141 = getelementptr inbounds [5 x i8], ptr %138, i64 %140
  %142 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %141) #4
  %143 = icmp eq i64 %142, 1
  br i1 %143, label %146, label %144

144:                                              ; preds = %135
  %145 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 117, ptr noundef nonnull @.str.42) #4
  tail call void @abort() #5
  unreachable

146:                                              ; preds = %135
  %147 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %148 = sext i32 %147 to i64
  %149 = getelementptr inbounds [5 x i8], ptr @b, i64 %148
  %150 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %151 = sext i32 %150 to i64
  %152 = getelementptr inbounds i8, ptr %149, i64 %151
  %153 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %152) #4
  %154 = icmp eq i64 %153, 1
  br i1 %154, label %157, label %155

155:                                              ; preds = %146
  %156 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 119, ptr noundef nonnull @.str.47) #4
  tail call void @abort() #5
  unreachable

157:                                              ; preds = %146
  %158 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %159 = sext i32 %158 to i64
  %160 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %159
  %161 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %162 = sext i32 %161 to i64
  %163 = getelementptr inbounds i8, ptr %160, i64 %162
  %164 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %163) #4
  %165 = icmp eq i64 %164, 2
  br i1 %165, label %168, label %166

166:                                              ; preds = %157
  %167 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 120, ptr noundef nonnull @.str.48) #4
  tail call void @abort() #5
  unreachable

168:                                              ; preds = %157
  %169 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %170 = sext i32 %169 to i64
  %171 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %170
  %172 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %173 = sext i32 %172 to i64
  %174 = getelementptr inbounds [5 x i8], ptr %171, i64 %173
  %175 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %174) #4
  %176 = icmp eq i64 %175, 4
  br i1 %176, label %179, label %177

177:                                              ; preds = %168
  %178 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 121, ptr noundef nonnull @.str.49) #4
  tail call void @abort() #5
  unreachable

179:                                              ; preds = %168
  %180 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %181 = sext i32 %180 to i64
  %182 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %181
  %183 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %184 = sext i32 %183 to i64
  %185 = getelementptr inbounds [5 x i8], ptr %182, i64 %184, i64 1
  %186 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %185) #4
  %187 = icmp eq i64 %186, 3
  br i1 %187, label %190, label %188

188:                                              ; preds = %179
  %189 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 122, ptr noundef nonnull @.str.50) #4
  tail call void @abort() #5
  unreachable

190:                                              ; preds = %179
  %191 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %192 = sext i32 %191 to i64
  %193 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %192
  %194 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %195 = sext i32 %194 to i64
  %196 = getelementptr inbounds [5 x i8], ptr %193, i64 %195, i64 2
  %197 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %196) #4
  %198 = icmp eq i64 %197, 2
  br i1 %198, label %201, label %199

199:                                              ; preds = %190
  %200 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 123, ptr noundef nonnull @.str.51) #4
  tail call void @abort() #5
  unreachable

201:                                              ; preds = %190
  %202 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %203 = sext i32 %202 to i64
  %204 = getelementptr inbounds [5 x i8], ptr @b, i64 %203
  %205 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %204) #4
  %206 = icmp eq i64 %205, 1
  br i1 %206, label %209, label %207

207:                                              ; preds = %201
  %208 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 129, ptr noundef nonnull @.str.52) #4
  tail call void @abort() #5
  unreachable

209:                                              ; preds = %201
  %210 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %211 = sext i32 %210 to i64
  %212 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %211
  %213 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %212) #4
  %214 = icmp eq i64 %213, 1
  br i1 %214, label %217, label %215

215:                                              ; preds = %209
  %216 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 130, ptr noundef nonnull @.str.53) #4
  tail call void @abort() #5
  unreachable

217:                                              ; preds = %209
  %218 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %219 = sext i32 %218 to i64
  %220 = getelementptr inbounds i8, ptr @b, i64 %219
  %221 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %220) #4
  %222 = icmp eq i64 %221, 1
  br i1 %222, label %225, label %223

223:                                              ; preds = %217
  %224 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 132, ptr noundef nonnull @.str.54) #4
  tail call void @abort() #5
  unreachable

225:                                              ; preds = %217
  %226 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %227 = sext i32 %226 to i64
  %228 = getelementptr inbounds [5 x i8], ptr @b, i64 %227
  %229 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %228) #4
  %230 = icmp eq i64 %229, 1
  br i1 %230, label %233, label %231

231:                                              ; preds = %225
  %232 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 133, ptr noundef nonnull @.str.55) #4
  tail call void @abort() #5
  unreachable

233:                                              ; preds = %225
  %234 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %235 = sext i32 %234 to i64
  %236 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %235
  %237 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %236) #4
  %238 = icmp eq i64 %237, 1
  br i1 %238, label %241, label %239

239:                                              ; preds = %233
  %240 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 134, ptr noundef nonnull @.str.56) #4
  tail call void @abort() #5
  unreachable

241:                                              ; preds = %233
  %242 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %243 = sext i32 %242 to i64
  %244 = getelementptr inbounds [5 x i8], ptr @b, i64 %243
  %245 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %246 = sext i32 %245 to i64
  %247 = getelementptr inbounds i8, ptr %244, i64 %246
  %248 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %247) #4
  %249 = icmp eq i64 %248, 1
  br i1 %249, label %252, label %250

250:                                              ; preds = %241
  %251 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 136, ptr noundef nonnull @.str.57) #4
  tail call void @abort() #5
  unreachable

252:                                              ; preds = %241
  %253 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %254 = sext i32 %253 to i64
  %255 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %254
  %256 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %257 = sext i32 %256 to i64
  %258 = getelementptr inbounds i8, ptr %255, i64 %257
  %259 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %258) #4
  %260 = icmp eq i64 %259, 1
  br i1 %260, label %263, label %261

261:                                              ; preds = %252
  %262 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 137, ptr noundef nonnull @.str.58) #4
  tail call void @abort() #5
  unreachable

263:                                              ; preds = %252
  %264 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %265 = sext i32 %264 to i64
  %266 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %265
  %267 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %268 = sext i32 %267 to i64
  %269 = getelementptr inbounds [5 x i8], ptr %266, i64 %268
  %270 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %269) #4
  %271 = icmp eq i64 %270, 1
  br i1 %271, label %274, label %272

272:                                              ; preds = %263
  %273 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 138, ptr noundef nonnull @.str.59) #4
  tail call void @abort() #5
  unreachable

274:                                              ; preds = %263
  %275 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %276 = sext i32 %275 to i64
  %277 = getelementptr inbounds [5 x i8], ptr @b, i64 %276
  %278 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %277) #4
  %279 = icmp eq i64 %278, 2
  br i1 %279, label %282, label %280

280:                                              ; preds = %274
  %281 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 140, ptr noundef nonnull @.str.60) #4
  tail call void @abort() #5
  unreachable

282:                                              ; preds = %274
  %283 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %284 = sext i32 %283 to i64
  %285 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %284
  %286 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %285) #4
  %287 = icmp eq i64 %286, 3
  br i1 %287, label %290, label %288

288:                                              ; preds = %282
  %289 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 141, ptr noundef nonnull @.str.61) #4
  tail call void @abort() #5
  unreachable

290:                                              ; preds = %282
  %291 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %292 = sext i32 %291 to i64
  %293 = getelementptr inbounds i8, ptr @b, i64 %292
  %294 = load i8, ptr %293, align 1
  %295 = icmp eq i8 %294, 0
  br i1 %295, label %298, label %296

296:                                              ; preds = %290
  %297 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 143, ptr noundef nonnull @.str.62) #4
  tail call void @abort() #5
  unreachable

298:                                              ; preds = %290
  %299 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %300 = sext i32 %299 to i64
  %301 = getelementptr inbounds [5 x i8], ptr @b, i64 %300
  %302 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %301) #4
  %303 = icmp eq i64 %302, 2
  br i1 %303, label %306, label %304

304:                                              ; preds = %298
  %305 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 144, ptr noundef nonnull @.str.63) #4
  tail call void @abort() #5
  unreachable

306:                                              ; preds = %298
  %307 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %308 = sext i32 %307 to i64
  %309 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %308
  %310 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %309) #4
  %311 = icmp eq i64 %310, 1
  br i1 %311, label %314, label %312

312:                                              ; preds = %306
  %313 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 145, ptr noundef nonnull @.str.56) #4
  tail call void @abort() #5
  unreachable

314:                                              ; preds = %306
  %315 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %316 = sext i32 %315 to i64
  %317 = getelementptr inbounds [5 x i8], ptr @b, i64 %316
  %318 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %319 = sext i32 %318 to i64
  %320 = getelementptr inbounds i8, ptr %317, i64 %319
  %321 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %320) #4
  %322 = icmp eq i64 %321, 1
  br i1 %322, label %325, label %323

323:                                              ; preds = %314
  %324 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 147, ptr noundef nonnull @.str.57) #4
  tail call void @abort() #5
  unreachable

325:                                              ; preds = %314
  %326 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %327 = sext i32 %326 to i64
  %328 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %327
  %329 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %330 = sext i32 %329 to i64
  %331 = getelementptr inbounds i8, ptr %328, i64 %330
  %332 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %331) #4
  %333 = icmp eq i64 %332, 1
  br i1 %333, label %336, label %334

334:                                              ; preds = %325
  %335 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 148, ptr noundef nonnull @.str.58) #4
  tail call void @abort() #5
  unreachable

336:                                              ; preds = %325
  %337 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %338 = sext i32 %337 to i64
  %339 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %338
  %340 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %341 = sext i32 %340 to i64
  %342 = getelementptr inbounds [5 x i8], ptr %339, i64 %341
  %343 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %342) #4
  %344 = icmp eq i64 %343, 1
  br i1 %344, label %347, label %345

345:                                              ; preds = %336
  %346 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 149, ptr noundef nonnull @.str.59) #4
  tail call void @abort() #5
  unreachable

347:                                              ; preds = %336
  %348 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %349 = sext i32 %348 to i64
  %350 = getelementptr inbounds [5 x i8], ptr @b, i64 %349
  %351 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %352 = sext i32 %351 to i64
  %353 = getelementptr inbounds i8, ptr %350, i64 %352
  %354 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %353) #4
  %355 = icmp eq i64 %354, 1
  br i1 %355, label %358, label %356

356:                                              ; preds = %347
  %357 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 151, ptr noundef nonnull @.str.64) #4
  tail call void @abort() #5
  unreachable

358:                                              ; preds = %347
  %359 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %360 = sext i32 %359 to i64
  %361 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %360
  %362 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %363 = sext i32 %362 to i64
  %364 = getelementptr inbounds i8, ptr %361, i64 %363
  %365 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %364) #4
  %366 = icmp eq i64 %365, 2
  br i1 %366, label %369, label %367

367:                                              ; preds = %358
  %368 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 152, ptr noundef nonnull @.str.65) #4
  tail call void @abort() #5
  unreachable

369:                                              ; preds = %358
  %370 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %371 = sext i32 %370 to i64
  %372 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %371
  %373 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %374 = sext i32 %373 to i64
  %375 = getelementptr inbounds [5 x i8], ptr %372, i64 %374
  %376 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %375) #4
  %377 = icmp eq i64 %376, 4
  br i1 %377, label %380, label %378

378:                                              ; preds = %369
  %379 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 153, ptr noundef nonnull @.str.66) #4
  tail call void @abort() #5
  unreachable

380:                                              ; preds = %369
  %381 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %382 = sext i32 %381 to i64
  %383 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %382
  %384 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %385 = sext i32 %384 to i64
  %386 = getelementptr inbounds [5 x i8], ptr %383, i64 %385, i64 1
  %387 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %386) #4
  %388 = icmp eq i64 %387, 3
  br i1 %388, label %391, label %389

389:                                              ; preds = %380
  %390 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 154, ptr noundef nonnull @.str.67) #4
  tail call void @abort() #5
  unreachable

391:                                              ; preds = %380
  %392 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %393 = sext i32 %392 to i64
  %394 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %393
  %395 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %396 = sext i32 %395 to i64
  %397 = getelementptr inbounds [5 x i8], ptr %394, i64 %396, i64 2
  %398 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %397) #4
  %399 = icmp eq i64 %398, 2
  br i1 %399, label %402, label %400

400:                                              ; preds = %391
  %401 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 155, ptr noundef nonnull @.str.68) #4
  tail call void @abort() #5
  unreachable

402:                                              ; preds = %391
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @test_array_off_2_2_5() local_unnamed_addr #0 {
  %1 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %2 = sext i32 %1 to i64
  %3 = getelementptr inbounds i8, ptr @b, i64 %2
  %4 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %3) #4
  %5 = icmp eq i64 %4, 1
  br i1 %5, label %8, label %6

6:                                                ; preds = %0
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 160, ptr noundef nonnull @.str.69) #4
  tail call void @abort() #5
  unreachable

8:                                                ; preds = %0
  %9 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds [5 x i8], ptr @b, i64 %10
  %12 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %13 = sext i32 %12 to i64
  %14 = getelementptr inbounds i8, ptr %11, i64 %13
  %15 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %14) #4
  %16 = icmp eq i64 %15, 1
  br i1 %16, label %19, label %17

17:                                               ; preds = %8
  %18 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 161, ptr noundef nonnull @.str.70) #4
  tail call void @abort() #5
  unreachable

19:                                               ; preds = %8
  %20 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %21 = sext i32 %20 to i64
  %22 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %21
  %23 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds i8, ptr %22, i64 %24
  %26 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %25) #4
  %27 = icmp eq i64 %26, 1
  br i1 %27, label %30, label %28

28:                                               ; preds = %19
  %29 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 162, ptr noundef nonnull @.str.71) #4
  tail call void @abort() #5
  unreachable

30:                                               ; preds = %19
  %31 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %32
  %34 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %35 = sext i32 %34 to i64
  %36 = getelementptr inbounds [5 x i8], ptr %33, i64 %35
  %37 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %38 = sext i32 %37 to i64
  %39 = getelementptr inbounds i8, ptr %36, i64 %38
  %40 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %39) #4
  %41 = icmp eq i64 %40, 1
  br i1 %41, label %44, label %42

42:                                               ; preds = %30
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 163, ptr noundef nonnull @.str.72) #4
  tail call void @abort() #5
  unreachable

44:                                               ; preds = %30
  %45 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %46 = sext i32 %45 to i64
  %47 = getelementptr inbounds i8, ptr @b, i64 %46
  %48 = load i8, ptr %47, align 1
  %49 = icmp eq i8 %48, 0
  br i1 %49, label %52, label %50

50:                                               ; preds = %44
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 165, ptr noundef nonnull @.str.73) #4
  tail call void @abort() #5
  unreachable

52:                                               ; preds = %44
  %53 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds [5 x i8], ptr @b, i64 %54
  %56 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %55) #4
  %57 = icmp eq i64 %56, 2
  br i1 %57, label %60, label %58

58:                                               ; preds = %52
  %59 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 166, ptr noundef nonnull @.str.74) #4
  tail call void @abort() #5
  unreachable

60:                                               ; preds = %52
  %61 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %62 = sext i32 %61 to i64
  %63 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %62
  %64 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %63) #4
  %65 = icmp eq i64 %64, 1
  br i1 %65, label %68, label %66

66:                                               ; preds = %60
  %67 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 167, ptr noundef nonnull @.str.75) #4
  tail call void @abort() #5
  unreachable

68:                                               ; preds = %60
  %69 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %70 = sext i32 %69 to i64
  %71 = getelementptr inbounds [5 x i8], ptr @b, i64 %70
  %72 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %73 = sext i32 %72 to i64
  %74 = getelementptr inbounds i8, ptr %71, i64 %73
  %75 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %74) #4
  %76 = icmp eq i64 %75, 1
  br i1 %76, label %79, label %77

77:                                               ; preds = %68
  %78 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 169, ptr noundef nonnull @.str.70) #4
  tail call void @abort() #5
  unreachable

79:                                               ; preds = %68
  %80 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %81 = sext i32 %80 to i64
  %82 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %81
  %83 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %84 = sext i32 %83 to i64
  %85 = getelementptr inbounds i8, ptr %82, i64 %84
  %86 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %85) #4
  %87 = icmp eq i64 %86, 1
  br i1 %87, label %90, label %88

88:                                               ; preds = %79
  %89 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 170, ptr noundef nonnull @.str.71) #4
  tail call void @abort() #5
  unreachable

90:                                               ; preds = %79
  %91 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %92 = sext i32 %91 to i64
  %93 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %92
  %94 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %95 = sext i32 %94 to i64
  %96 = getelementptr inbounds [5 x i8], ptr %93, i64 %95
  %97 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %96) #4
  %98 = icmp eq i64 %97, 1
  br i1 %98, label %101, label %99

99:                                               ; preds = %90
  %100 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 171, ptr noundef nonnull @.str.76) #4
  tail call void @abort() #5
  unreachable

101:                                              ; preds = %90
  %102 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %103 = sext i32 %102 to i64
  %104 = getelementptr inbounds [5 x i8], ptr @b, i64 %103
  %105 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %106 = sext i32 %105 to i64
  %107 = getelementptr inbounds i8, ptr %104, i64 %106
  %108 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %107) #4
  %109 = icmp eq i64 %108, 1
  br i1 %109, label %112, label %110

110:                                              ; preds = %101
  %111 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 173, ptr noundef nonnull @.str.77) #4
  tail call void @abort() #5
  unreachable

112:                                              ; preds = %101
  %113 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %114 = sext i32 %113 to i64
  %115 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %114
  %116 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %117 = sext i32 %116 to i64
  %118 = getelementptr inbounds i8, ptr %115, i64 %117
  %119 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %118) #4
  %120 = icmp eq i64 %119, 2
  br i1 %120, label %123, label %121

121:                                              ; preds = %112
  %122 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 174, ptr noundef nonnull @.str.78) #4
  tail call void @abort() #5
  unreachable

123:                                              ; preds = %112
  %124 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %125 = sext i32 %124 to i64
  %126 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %125
  %127 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %128 = sext i32 %127 to i64
  %129 = getelementptr inbounds [5 x i8], ptr %126, i64 %128
  %130 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %129) #4
  %131 = icmp eq i64 %130, 4
  br i1 %131, label %134, label %132

132:                                              ; preds = %123
  %133 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 175, ptr noundef nonnull @.str.79) #4
  tail call void @abort() #5
  unreachable

134:                                              ; preds = %123
  %135 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %136 = sext i32 %135 to i64
  %137 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %136
  %138 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %139 = sext i32 %138 to i64
  %140 = getelementptr inbounds [5 x i8], ptr %137, i64 %139, i64 1
  %141 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %140) #4
  %142 = icmp eq i64 %141, 3
  br i1 %142, label %145, label %143

143:                                              ; preds = %134
  %144 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 176, ptr noundef nonnull @.str.80) #4
  tail call void @abort() #5
  unreachable

145:                                              ; preds = %134
  %146 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %147 = sext i32 %146 to i64
  %148 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %147
  %149 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %150 = sext i32 %149 to i64
  %151 = getelementptr inbounds [5 x i8], ptr %148, i64 %150, i64 2
  %152 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %151) #4
  %153 = icmp eq i64 %152, 2
  br i1 %153, label %156, label %154

154:                                              ; preds = %145
  %155 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 177, ptr noundef nonnull @.str.81) #4
  tail call void @abort() #5
  unreachable

156:                                              ; preds = %145
  %157 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %158 = sext i32 %157 to i64
  %159 = getelementptr inbounds i8, ptr @b, i64 %158
  %160 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %159) #4
  %161 = icmp eq i64 %160, 1
  br i1 %161, label %164, label %162

162:                                              ; preds = %156
  %163 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 183, ptr noundef nonnull @.str.82) #4
  tail call void @abort() #5
  unreachable

164:                                              ; preds = %156
  %165 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %166 = sext i32 %165 to i64
  %167 = getelementptr inbounds [5 x i8], ptr @b, i64 %166
  %168 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %169 = sext i32 %168 to i64
  %170 = getelementptr inbounds i8, ptr %167, i64 %169
  %171 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %170) #4
  %172 = icmp eq i64 %171, 1
  br i1 %172, label %175, label %173

173:                                              ; preds = %164
  %174 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 184, ptr noundef nonnull @.str.83) #4
  tail call void @abort() #5
  unreachable

175:                                              ; preds = %164
  %176 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %177 = sext i32 %176 to i64
  %178 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %177
  %179 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %180 = sext i32 %179 to i64
  %181 = getelementptr inbounds i8, ptr %178, i64 %180
  %182 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %181) #4
  %183 = icmp eq i64 %182, 1
  br i1 %183, label %186, label %184

184:                                              ; preds = %175
  %185 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 185, ptr noundef nonnull @.str.84) #4
  tail call void @abort() #5
  unreachable

186:                                              ; preds = %175
  %187 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %188 = sext i32 %187 to i64
  %189 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %188
  %190 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %191 = sext i32 %190 to i64
  %192 = getelementptr inbounds [5 x i8], ptr %189, i64 %191
  %193 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %194 = sext i32 %193 to i64
  %195 = getelementptr inbounds i8, ptr %192, i64 %194
  %196 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %195) #4
  %197 = icmp eq i64 %196, 1
  br i1 %197, label %200, label %198

198:                                              ; preds = %186
  %199 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 186, ptr noundef nonnull @.str.72) #4
  tail call void @abort() #5
  unreachable

200:                                              ; preds = %186
  %201 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %202 = sext i32 %201 to i64
  %203 = getelementptr inbounds i8, ptr @b, i64 %202
  %204 = load i8, ptr %203, align 1
  %205 = icmp eq i8 %204, 0
  br i1 %205, label %208, label %206

206:                                              ; preds = %200
  %207 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 188, ptr noundef nonnull @.str.85) #4
  tail call void @abort() #5
  unreachable

208:                                              ; preds = %200
  %209 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %210 = sext i32 %209 to i64
  %211 = getelementptr inbounds [5 x i8], ptr @b, i64 %210
  %212 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %211) #4
  %213 = icmp eq i64 %212, 2
  br i1 %213, label %216, label %214

214:                                              ; preds = %208
  %215 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 189, ptr noundef nonnull @.str.86) #4
  tail call void @abort() #5
  unreachable

216:                                              ; preds = %208
  %217 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %218 = sext i32 %217 to i64
  %219 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %218
  %220 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %219) #4
  %221 = icmp eq i64 %220, 1
  br i1 %221, label %224, label %222

222:                                              ; preds = %216
  %223 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 190, ptr noundef nonnull @.str.87) #4
  tail call void @abort() #5
  unreachable

224:                                              ; preds = %216
  %225 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %226 = sext i32 %225 to i64
  %227 = getelementptr inbounds [5 x i8], ptr @b, i64 %226
  %228 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %229 = sext i32 %228 to i64
  %230 = getelementptr inbounds i8, ptr %227, i64 %229
  %231 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %230) #4
  %232 = icmp eq i64 %231, 1
  br i1 %232, label %235, label %233

233:                                              ; preds = %224
  %234 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 192, ptr noundef nonnull @.str.83) #4
  tail call void @abort() #5
  unreachable

235:                                              ; preds = %224
  %236 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %237 = sext i32 %236 to i64
  %238 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %237
  %239 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %240 = sext i32 %239 to i64
  %241 = getelementptr inbounds i8, ptr %238, i64 %240
  %242 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %241) #4
  %243 = icmp eq i64 %242, 1
  br i1 %243, label %246, label %244

244:                                              ; preds = %235
  %245 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 193, ptr noundef nonnull @.str.84) #4
  tail call void @abort() #5
  unreachable

246:                                              ; preds = %235
  %247 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %248 = sext i32 %247 to i64
  %249 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %248
  %250 = load volatile i32, ptr @v0, align 4, !tbaa !6
  %251 = sext i32 %250 to i64
  %252 = getelementptr inbounds [5 x i8], ptr %249, i64 %251
  %253 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %252) #4
  %254 = icmp eq i64 %253, 1
  br i1 %254, label %257, label %255

255:                                              ; preds = %246
  %256 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 194, ptr noundef nonnull @.str.88) #4
  tail call void @abort() #5
  unreachable

257:                                              ; preds = %246
  %258 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %259 = sext i32 %258 to i64
  %260 = getelementptr inbounds [5 x i8], ptr @b, i64 %259
  %261 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %262 = sext i32 %261 to i64
  %263 = getelementptr inbounds i8, ptr %260, i64 %262
  %264 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %263) #4
  %265 = icmp eq i64 %264, 1
  br i1 %265, label %268, label %266

266:                                              ; preds = %257
  %267 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 196, ptr noundef nonnull @.str.89) #4
  tail call void @abort() #5
  unreachable

268:                                              ; preds = %257
  %269 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %270 = sext i32 %269 to i64
  %271 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %270
  %272 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %273 = sext i32 %272 to i64
  %274 = getelementptr inbounds i8, ptr %271, i64 %273
  %275 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %274) #4
  %276 = icmp eq i64 %275, 2
  br i1 %276, label %279, label %277

277:                                              ; preds = %268
  %278 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 197, ptr noundef nonnull @.str.90) #4
  tail call void @abort() #5
  unreachable

279:                                              ; preds = %268
  %280 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %281 = sext i32 %280 to i64
  %282 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %281
  %283 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %284 = sext i32 %283 to i64
  %285 = getelementptr inbounds [5 x i8], ptr %282, i64 %284
  %286 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %285) #4
  %287 = icmp eq i64 %286, 4
  br i1 %287, label %290, label %288

288:                                              ; preds = %279
  %289 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 198, ptr noundef nonnull @.str.91) #4
  tail call void @abort() #5
  unreachable

290:                                              ; preds = %279
  %291 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %292 = sext i32 %291 to i64
  %293 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %292
  %294 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %295 = sext i32 %294 to i64
  %296 = getelementptr inbounds [5 x i8], ptr %293, i64 %295, i64 1
  %297 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %296) #4
  %298 = icmp eq i64 %297, 3
  br i1 %298, label %301, label %299

299:                                              ; preds = %290
  %300 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 199, ptr noundef nonnull @.str.92) #4
  tail call void @abort() #5
  unreachable

301:                                              ; preds = %290
  %302 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %303 = sext i32 %302 to i64
  %304 = getelementptr inbounds [2 x [5 x i8]], ptr @b, i64 %303
  %305 = load volatile i32, ptr @v1, align 4, !tbaa !6
  %306 = sext i32 %305 to i64
  %307 = getelementptr inbounds [5 x i8], ptr %304, i64 %306, i64 2
  %308 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %307) #4
  %309 = icmp eq i64 %308, 2
  br i1 %309, label %312, label %310

310:                                              ; preds = %301
  %311 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 200, ptr noundef nonnull @.str.93) #4
  tail call void @abort() #5
  unreachable

312:                                              ; preds = %301
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void @test_array_ref_2_3()
  tail call void @test_array_off_2_3()
  tail call void @test_array_ref_2_2_5()
  tail call void @test_array_off_2_2_5()
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

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
