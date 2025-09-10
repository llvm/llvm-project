; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/matrix.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/matrix.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [22 x i8] c"get_gcd(%d, %d) = %d\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @mysort(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) initializes((0, 32)) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr %0, align 4, !tbaa !6
  store i32 %3, ptr %1, align 4, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %5 = load i32, ptr %4, align 4, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 4
  store i32 %5, ptr %6, align 4, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %8 = load i32, ptr %7, align 4, !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 %8, ptr %9, align 4, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 12
  %11 = load i32, ptr %10, align 4, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 12
  store i32 %11, ptr %12, align 4, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %14 = load i32, ptr %13, align 4, !tbaa !6
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i32 %14, ptr %15, align 4, !tbaa !6
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %17 = load i32, ptr %16, align 4, !tbaa !6
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 20
  store i32 %17, ptr %18, align 4, !tbaa !6
  %19 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %20 = load i32, ptr %19, align 4, !tbaa !6
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store i32 %20, ptr %21, align 4, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %0, i64 28
  %23 = load i32, ptr %22, align 4, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 28
  store i32 %23, ptr %24, align 4, !tbaa !6
  %25 = icmp sgt i32 %5, %3
  br i1 %25, label %26, label %27

26:                                               ; preds = %2
  store i32 %5, ptr %1, align 4, !tbaa !6
  store i32 %3, ptr %6, align 4, !tbaa !6
  br label %27

27:                                               ; preds = %26, %2
  %28 = phi i32 [ %3, %26 ], [ %5, %2 ]
  %29 = phi i32 [ %5, %26 ], [ %3, %2 ]
  %30 = icmp sgt i32 %8, %29
  br i1 %30, label %31, label %32

31:                                               ; preds = %27
  store i32 %8, ptr %1, align 4, !tbaa !6
  store i32 %29, ptr %9, align 4, !tbaa !6
  br label %32

32:                                               ; preds = %31, %27
  %33 = phi i32 [ %29, %31 ], [ %8, %27 ]
  %34 = phi i32 [ %8, %31 ], [ %29, %27 ]
  %35 = icmp sgt i32 %11, %34
  br i1 %35, label %36, label %37

36:                                               ; preds = %32
  store i32 %11, ptr %1, align 4, !tbaa !6
  store i32 %34, ptr %12, align 4, !tbaa !6
  br label %37

37:                                               ; preds = %36, %32
  %38 = phi i32 [ %34, %36 ], [ %11, %32 ]
  %39 = phi i32 [ %11, %36 ], [ %34, %32 ]
  %40 = icmp sgt i32 %14, %39
  br i1 %40, label %41, label %42

41:                                               ; preds = %37
  store i32 %14, ptr %1, align 4, !tbaa !6
  store i32 %39, ptr %15, align 4, !tbaa !6
  br label %42

42:                                               ; preds = %41, %37
  %43 = phi i32 [ %39, %41 ], [ %14, %37 ]
  %44 = phi i32 [ %14, %41 ], [ %39, %37 ]
  %45 = icmp sgt i32 %17, %44
  br i1 %45, label %46, label %47

46:                                               ; preds = %42
  store i32 %17, ptr %1, align 4, !tbaa !6
  store i32 %44, ptr %18, align 4, !tbaa !6
  br label %47

47:                                               ; preds = %46, %42
  %48 = phi i32 [ %44, %46 ], [ %17, %42 ]
  %49 = phi i32 [ %17, %46 ], [ %44, %42 ]
  %50 = icmp sgt i32 %20, %49
  br i1 %50, label %51, label %52

51:                                               ; preds = %47
  store i32 %20, ptr %1, align 4, !tbaa !6
  store i32 %49, ptr %21, align 4, !tbaa !6
  br label %52

52:                                               ; preds = %51, %47
  %53 = phi i32 [ %49, %51 ], [ %20, %47 ]
  %54 = phi i32 [ %20, %51 ], [ %49, %47 ]
  %55 = icmp sgt i32 %23, %54
  br i1 %55, label %56, label %57

56:                                               ; preds = %52
  store i32 %23, ptr %1, align 4, !tbaa !6
  store i32 %54, ptr %24, align 4, !tbaa !6
  br label %57

57:                                               ; preds = %52, %56
  %58 = phi i32 [ %54, %56 ], [ %23, %52 ]
  %59 = icmp sgt i32 %33, %28
  br i1 %59, label %60, label %61

60:                                               ; preds = %57
  store i32 %33, ptr %6, align 4, !tbaa !6
  store i32 %28, ptr %9, align 4, !tbaa !6
  br label %61

61:                                               ; preds = %60, %57
  %62 = phi i32 [ %28, %60 ], [ %33, %57 ]
  %63 = phi i32 [ %33, %60 ], [ %28, %57 ]
  %64 = icmp sgt i32 %38, %63
  br i1 %64, label %65, label %66

65:                                               ; preds = %61
  store i32 %38, ptr %6, align 4, !tbaa !6
  store i32 %63, ptr %12, align 4, !tbaa !6
  br label %66

66:                                               ; preds = %65, %61
  %67 = phi i32 [ %63, %65 ], [ %38, %61 ]
  %68 = phi i32 [ %38, %65 ], [ %63, %61 ]
  %69 = icmp sgt i32 %43, %68
  br i1 %69, label %70, label %71

70:                                               ; preds = %66
  store i32 %43, ptr %6, align 4, !tbaa !6
  store i32 %68, ptr %15, align 4, !tbaa !6
  br label %71

71:                                               ; preds = %70, %66
  %72 = phi i32 [ %68, %70 ], [ %43, %66 ]
  %73 = phi i32 [ %43, %70 ], [ %68, %66 ]
  %74 = icmp sgt i32 %48, %73
  br i1 %74, label %75, label %76

75:                                               ; preds = %71
  store i32 %48, ptr %6, align 4, !tbaa !6
  store i32 %73, ptr %18, align 4, !tbaa !6
  br label %76

76:                                               ; preds = %75, %71
  %77 = phi i32 [ %73, %75 ], [ %48, %71 ]
  %78 = phi i32 [ %48, %75 ], [ %73, %71 ]
  %79 = icmp sgt i32 %53, %78
  br i1 %79, label %80, label %81

80:                                               ; preds = %76
  store i32 %53, ptr %6, align 4, !tbaa !6
  store i32 %78, ptr %21, align 4, !tbaa !6
  br label %81

81:                                               ; preds = %80, %76
  %82 = phi i32 [ %78, %80 ], [ %53, %76 ]
  %83 = phi i32 [ %53, %80 ], [ %78, %76 ]
  %84 = icmp sgt i32 %58, %83
  br i1 %84, label %85, label %86

85:                                               ; preds = %81
  store i32 %58, ptr %6, align 4, !tbaa !6
  store i32 %83, ptr %24, align 4, !tbaa !6
  br label %86

86:                                               ; preds = %81, %85
  %87 = phi i32 [ %83, %85 ], [ %58, %81 ]
  %88 = icmp sgt i32 %67, %62
  br i1 %88, label %89, label %90

89:                                               ; preds = %86
  store i32 %67, ptr %9, align 4, !tbaa !6
  store i32 %62, ptr %12, align 4, !tbaa !6
  br label %90

90:                                               ; preds = %89, %86
  %91 = phi i32 [ %62, %89 ], [ %67, %86 ]
  %92 = phi i32 [ %67, %89 ], [ %62, %86 ]
  %93 = icmp sgt i32 %72, %92
  br i1 %93, label %94, label %95

94:                                               ; preds = %90
  store i32 %72, ptr %9, align 4, !tbaa !6
  store i32 %92, ptr %15, align 4, !tbaa !6
  br label %95

95:                                               ; preds = %94, %90
  %96 = phi i32 [ %92, %94 ], [ %72, %90 ]
  %97 = phi i32 [ %72, %94 ], [ %92, %90 ]
  %98 = icmp sgt i32 %77, %97
  br i1 %98, label %99, label %100

99:                                               ; preds = %95
  store i32 %77, ptr %9, align 4, !tbaa !6
  store i32 %97, ptr %18, align 4, !tbaa !6
  br label %100

100:                                              ; preds = %99, %95
  %101 = phi i32 [ %97, %99 ], [ %77, %95 ]
  %102 = phi i32 [ %77, %99 ], [ %97, %95 ]
  %103 = icmp sgt i32 %82, %102
  br i1 %103, label %104, label %105

104:                                              ; preds = %100
  store i32 %82, ptr %9, align 4, !tbaa !6
  store i32 %102, ptr %21, align 4, !tbaa !6
  br label %105

105:                                              ; preds = %104, %100
  %106 = phi i32 [ %102, %104 ], [ %82, %100 ]
  %107 = phi i32 [ %82, %104 ], [ %102, %100 ]
  %108 = icmp sgt i32 %87, %107
  br i1 %108, label %109, label %110

109:                                              ; preds = %105
  store i32 %87, ptr %9, align 4, !tbaa !6
  store i32 %107, ptr %24, align 4, !tbaa !6
  br label %110

110:                                              ; preds = %105, %109
  %111 = phi i32 [ %107, %109 ], [ %87, %105 ]
  %112 = icmp sgt i32 %96, %91
  br i1 %112, label %113, label %114

113:                                              ; preds = %110
  store i32 %96, ptr %12, align 4, !tbaa !6
  store i32 %91, ptr %15, align 4, !tbaa !6
  br label %114

114:                                              ; preds = %113, %110
  %115 = phi i32 [ %91, %113 ], [ %96, %110 ]
  %116 = phi i32 [ %96, %113 ], [ %91, %110 ]
  %117 = icmp sgt i32 %101, %116
  br i1 %117, label %118, label %119

118:                                              ; preds = %114
  store i32 %101, ptr %12, align 4, !tbaa !6
  store i32 %116, ptr %18, align 4, !tbaa !6
  br label %119

119:                                              ; preds = %118, %114
  %120 = phi i32 [ %116, %118 ], [ %101, %114 ]
  %121 = phi i32 [ %101, %118 ], [ %116, %114 ]
  %122 = icmp sgt i32 %106, %121
  br i1 %122, label %123, label %124

123:                                              ; preds = %119
  store i32 %106, ptr %12, align 4, !tbaa !6
  store i32 %121, ptr %21, align 4, !tbaa !6
  br label %124

124:                                              ; preds = %123, %119
  %125 = phi i32 [ %121, %123 ], [ %106, %119 ]
  %126 = phi i32 [ %106, %123 ], [ %121, %119 ]
  %127 = icmp sgt i32 %111, %126
  br i1 %127, label %128, label %129

128:                                              ; preds = %124
  store i32 %111, ptr %12, align 4, !tbaa !6
  store i32 %126, ptr %24, align 4, !tbaa !6
  br label %129

129:                                              ; preds = %124, %128
  %130 = phi i32 [ %126, %128 ], [ %111, %124 ]
  %131 = icmp sgt i32 %120, %115
  br i1 %131, label %132, label %133

132:                                              ; preds = %129
  store i32 %120, ptr %15, align 4, !tbaa !6
  store i32 %115, ptr %18, align 4, !tbaa !6
  br label %133

133:                                              ; preds = %132, %129
  %134 = phi i32 [ %115, %132 ], [ %120, %129 ]
  %135 = phi i32 [ %120, %132 ], [ %115, %129 ]
  %136 = icmp sgt i32 %125, %135
  br i1 %136, label %137, label %138

137:                                              ; preds = %133
  store i32 %125, ptr %15, align 4, !tbaa !6
  store i32 %135, ptr %21, align 4, !tbaa !6
  br label %138

138:                                              ; preds = %137, %133
  %139 = phi i32 [ %135, %137 ], [ %125, %133 ]
  %140 = phi i32 [ %125, %137 ], [ %135, %133 ]
  %141 = icmp sgt i32 %130, %140
  br i1 %141, label %142, label %143

142:                                              ; preds = %138
  store i32 %130, ptr %15, align 4, !tbaa !6
  store i32 %140, ptr %24, align 4, !tbaa !6
  br label %143

143:                                              ; preds = %138, %142
  %144 = phi i32 [ %140, %142 ], [ %130, %138 ]
  %145 = icmp sgt i32 %139, %134
  br i1 %145, label %146, label %147

146:                                              ; preds = %143
  store i32 %139, ptr %18, align 4, !tbaa !6
  store i32 %134, ptr %21, align 4, !tbaa !6
  br label %147

147:                                              ; preds = %146, %143
  %148 = phi i32 [ %134, %146 ], [ %139, %143 ]
  %149 = phi i32 [ %139, %146 ], [ %134, %143 ]
  %150 = icmp sgt i32 %144, %149
  br i1 %150, label %151, label %152

151:                                              ; preds = %147
  store i32 %144, ptr %18, align 4, !tbaa !6
  store i32 %149, ptr %24, align 4, !tbaa !6
  br label %152

152:                                              ; preds = %147, %151
  %153 = phi i32 [ %149, %151 ], [ %144, %147 ]
  %154 = icmp sgt i32 %153, %148
  br i1 %154, label %155, label %156

155:                                              ; preds = %152
  store i32 %153, ptr %21, align 4, !tbaa !6
  store i32 %148, ptr %24, align 4, !tbaa !6
  br label %156

156:                                              ; preds = %152, %155
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree norecurse nosync nounwind memory(none) uwtable
define dso_local i32 @get_gcd(i32 noundef %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %9, label %4

4:                                                ; preds = %2, %4
  %5 = phi i32 [ %7, %4 ], [ %1, %2 ]
  %6 = phi i32 [ %5, %4 ], [ %0, %2 ]
  %7 = srem i32 %6, %5
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %4

9:                                                ; preds = %4, %2
  %10 = phi i32 [ %0, %2 ], [ %5, %4 ]
  ret i32 %10
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @my_test(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #3 {
  %3 = alloca [8 x [8 x i32]], align 4
  %4 = alloca [8 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #5
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 4
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 240
  br label %21

21:                                               ; preds = %2, %139
  %22 = phi i64 [ 0, %2 ], [ %142, %139 ]
  %23 = getelementptr inbounds nuw [8 x i32], ptr %3, i64 %22
  %24 = getelementptr inbounds nuw [8 x i32], ptr %0, i64 %22
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 4
  %26 = getelementptr inbounds nuw i8, ptr %24, i64 8
  %27 = getelementptr inbounds nuw i8, ptr %24, i64 12
  %28 = getelementptr inbounds nuw i8, ptr %24, i64 16
  %29 = getelementptr inbounds nuw i8, ptr %24, i64 20
  %30 = getelementptr inbounds nuw i8, ptr %24, i64 24
  %31 = getelementptr inbounds nuw i8, ptr %24, i64 28
  %32 = load <2 x i32>, ptr %24, align 4, !tbaa !6
  %33 = load <2 x i32>, ptr %26, align 4, !tbaa !6
  %34 = load <2 x i32>, ptr %28, align 4, !tbaa !6
  %35 = load <2 x i32>, ptr %30, align 4, !tbaa !6
  %36 = load <4 x i32>, ptr %1, align 4, !tbaa !6
  %37 = load <4 x i32>, ptr %6, align 4, !tbaa !6
  %38 = load <4 x i32>, ptr %7, align 4, !tbaa !6
  %39 = load <4 x i32>, ptr %8, align 4, !tbaa !6
  %40 = load <4 x i32>, ptr %9, align 4, !tbaa !6
  %41 = load <4 x i32>, ptr %10, align 4, !tbaa !6
  %42 = load <4 x i32>, ptr %11, align 4, !tbaa !6
  %43 = load <4 x i32>, ptr %12, align 4, !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %23, i64 16
  %45 = load i32, ptr %24, align 4, !tbaa !6
  %46 = load i32, ptr %25, align 4, !tbaa !6
  %47 = load i32, ptr %26, align 4, !tbaa !6
  %48 = load i32, ptr %27, align 4, !tbaa !6
  %49 = load i32, ptr %28, align 4, !tbaa !6
  %50 = load i32, ptr %29, align 4, !tbaa !6
  %51 = load i32, ptr %30, align 4, !tbaa !6
  %52 = load i32, ptr %31, align 4, !tbaa !6
  %53 = load <2 x i32>, ptr %24, align 4, !tbaa !6
  %54 = load <2 x i32>, ptr %26, align 4, !tbaa !6
  %55 = load <2 x i32>, ptr %28, align 4, !tbaa !6
  %56 = load <2 x i32>, ptr %30, align 4, !tbaa !6
  %57 = shufflevector <2 x i32> %32, <2 x i32> %53, <2 x i32> <i32 0, i32 2>
  %58 = shufflevector <2 x i32> %57, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  %59 = mul nsw <4 x i32> %36, %58
  %60 = shufflevector <2 x i32> %32, <2 x i32> %53, <2 x i32> <i32 1, i32 3>
  %61 = shufflevector <2 x i32> %60, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  %62 = mul nsw <4 x i32> %37, %61
  %63 = add nsw <4 x i32> %59, %62
  %64 = shufflevector <2 x i32> %33, <2 x i32> %54, <2 x i32> <i32 0, i32 2>
  %65 = shufflevector <2 x i32> %64, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  %66 = mul nsw <4 x i32> %38, %65
  %67 = add nsw <4 x i32> %63, %66
  %68 = shufflevector <2 x i32> %33, <2 x i32> %54, <2 x i32> <i32 1, i32 3>
  %69 = shufflevector <2 x i32> %68, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  %70 = mul nsw <4 x i32> %39, %69
  %71 = add nsw <4 x i32> %67, %70
  %72 = shufflevector <2 x i32> %34, <2 x i32> %55, <2 x i32> <i32 0, i32 2>
  %73 = shufflevector <2 x i32> %72, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  %74 = mul nsw <4 x i32> %40, %73
  %75 = add nsw <4 x i32> %71, %74
  %76 = shufflevector <2 x i32> %34, <2 x i32> %55, <2 x i32> <i32 1, i32 3>
  %77 = shufflevector <2 x i32> %76, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  %78 = mul nsw <4 x i32> %41, %77
  %79 = add nsw <4 x i32> %75, %78
  %80 = shufflevector <2 x i32> %35, <2 x i32> %56, <2 x i32> <i32 0, i32 2>
  %81 = shufflevector <2 x i32> %80, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  %82 = mul nsw <4 x i32> %42, %81
  %83 = add nsw <4 x i32> %79, %82
  %84 = shufflevector <2 x i32> %35, <2 x i32> %56, <2 x i32> <i32 1, i32 3>
  %85 = shufflevector <2 x i32> %84, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 0, i32 1>
  %86 = mul nsw <4 x i32> %43, %85
  %87 = add nsw <4 x i32> %83, %86
  store <4 x i32> %87, ptr %23, align 4, !tbaa !6
  %88 = load <4 x i32>, ptr %13, align 4, !tbaa !6
  %89 = insertelement <2 x i32> %53, i32 %45, i64 1
  %90 = shufflevector <2 x i32> %89, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %91 = mul nsw <4 x i32> %88, %90
  %92 = load <4 x i32>, ptr %14, align 4, !tbaa !6
  %93 = shufflevector <2 x i32> %53, <2 x i32> poison, <2 x i32> <i32 1, i32 poison>
  %94 = insertelement <2 x i32> %93, i32 %46, i64 1
  %95 = shufflevector <2 x i32> %94, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %96 = mul nsw <4 x i32> %92, %95
  %97 = add nsw <4 x i32> %91, %96
  %98 = load <4 x i32>, ptr %15, align 4, !tbaa !6
  %99 = insertelement <2 x i32> %54, i32 %47, i64 1
  %100 = shufflevector <2 x i32> %99, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %101 = mul nsw <4 x i32> %98, %100
  %102 = add nsw <4 x i32> %97, %101
  %103 = load <4 x i32>, ptr %16, align 4, !tbaa !6
  %104 = shufflevector <2 x i32> %54, <2 x i32> poison, <2 x i32> <i32 1, i32 poison>
  %105 = insertelement <2 x i32> %104, i32 %48, i64 1
  %106 = shufflevector <2 x i32> %105, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %107 = mul nsw <4 x i32> %103, %106
  %108 = add nsw <4 x i32> %102, %107
  %109 = load <4 x i32>, ptr %17, align 4, !tbaa !6
  %110 = insertelement <2 x i32> %55, i32 %49, i64 1
  %111 = shufflevector <2 x i32> %110, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %112 = mul nsw <4 x i32> %109, %111
  %113 = add nsw <4 x i32> %108, %112
  %114 = load <4 x i32>, ptr %18, align 4, !tbaa !6
  %115 = shufflevector <2 x i32> %55, <2 x i32> poison, <2 x i32> <i32 1, i32 poison>
  %116 = insertelement <2 x i32> %115, i32 %50, i64 1
  %117 = shufflevector <2 x i32> %116, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %118 = mul nsw <4 x i32> %114, %117
  %119 = add nsw <4 x i32> %113, %118
  %120 = load <4 x i32>, ptr %19, align 4, !tbaa !6
  %121 = insertelement <2 x i32> %56, i32 %51, i64 1
  %122 = shufflevector <2 x i32> %121, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %123 = mul nsw <4 x i32> %120, %122
  %124 = add nsw <4 x i32> %119, %123
  %125 = load <4 x i32>, ptr %20, align 4, !tbaa !6
  %126 = shufflevector <2 x i32> %56, <2 x i32> poison, <2 x i32> <i32 1, i32 poison>
  %127 = insertelement <2 x i32> %126, i32 %52, i64 1
  %128 = shufflevector <2 x i32> %127, <2 x i32> poison, <4 x i32> <i32 0, i32 0, i32 1, i32 1>
  %129 = mul nsw <4 x i32> %125, %128
  %130 = add nsw <4 x i32> %124, %129
  store <4 x i32> %130, ptr %44, align 4, !tbaa !6
  call void @mysort(ptr noundef nonnull %23, ptr noundef nonnull %4)
  %131 = load i32, ptr %4, align 4, !tbaa !6
  %132 = load i32, ptr %5, align 4, !tbaa !6
  %133 = icmp eq i32 %132, 0
  br i1 %133, label %139, label %134

134:                                              ; preds = %21, %134
  %135 = phi i32 [ %137, %134 ], [ %132, %21 ]
  %136 = phi i32 [ %135, %134 ], [ %131, %21 ]
  %137 = srem i32 %136, %135
  %138 = icmp eq i32 %137, 0
  br i1 %138, label %139, label %134

139:                                              ; preds = %134, %21
  %140 = phi i32 [ %131, %21 ], [ %135, %134 ]
  %141 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %131, i32 noundef %132, i32 noundef %140)
  %142 = add nuw nsw i64 %22, 1
  %143 = icmp eq i64 %142, 8
  br i1 %143, label %144, label %21

144:                                              ; preds = %139
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 {
  %1 = alloca [8 x [8 x i32]], align 16
  %2 = alloca [8 x [8 x i32]], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  store <4 x i32> <i32 79, i32 80, i32 81, i32 82>, ptr %1, align 16, !tbaa !6
  store <4 x i32> <i32 -255, i32 -256, i32 -257, i32 -258>, ptr %2, align 16, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store <4 x i32> <i32 83, i32 84, i32 85, i32 86>, ptr %3, align 16, !tbaa !6
  store <4 x i32> <i32 -259, i32 -260, i32 -261, i32 -262>, ptr %4, align 16, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 32
  store <4 x i32> <i32 158, i32 160, i32 162, i32 164>, ptr %5, align 16, !tbaa !6
  store <4 x i32> zeroinitializer, ptr %6, align 16, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 48
  store <4 x i32> <i32 166, i32 168, i32 170, i32 172>, ptr %7, align 16, !tbaa !6
  store <4 x i32> zeroinitializer, ptr %8, align 16, !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 64
  store <4 x i32> <i32 237, i32 240, i32 243, i32 246>, ptr %9, align 16, !tbaa !6
  store <4 x i32> <i32 255, i32 256, i32 257, i32 258>, ptr %10, align 16, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 80
  store <4 x i32> <i32 249, i32 252, i32 255, i32 258>, ptr %11, align 16, !tbaa !6
  store <4 x i32> <i32 259, i32 260, i32 261, i32 262>, ptr %12, align 16, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %14 = getelementptr inbounds nuw i8, ptr %2, i64 96
  store <4 x i32> <i32 316, i32 320, i32 324, i32 328>, ptr %13, align 16, !tbaa !6
  store <4 x i32> <i32 510, i32 512, i32 514, i32 516>, ptr %14, align 16, !tbaa !6
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %16 = getelementptr inbounds nuw i8, ptr %2, i64 112
  store <4 x i32> <i32 332, i32 336, i32 340, i32 344>, ptr %15, align 16, !tbaa !6
  store <4 x i32> <i32 518, i32 520, i32 522, i32 524>, ptr %16, align 16, !tbaa !6
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 128
  %18 = getelementptr inbounds nuw i8, ptr %2, i64 128
  store <4 x i32> <i32 395, i32 400, i32 405, i32 410>, ptr %17, align 16, !tbaa !6
  store <4 x i32> <i32 765, i32 768, i32 771, i32 774>, ptr %18, align 16, !tbaa !6
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 144
  %20 = getelementptr inbounds nuw i8, ptr %2, i64 144
  store <4 x i32> <i32 415, i32 420, i32 425, i32 430>, ptr %19, align 16, !tbaa !6
  store <4 x i32> <i32 777, i32 780, i32 783, i32 786>, ptr %20, align 16, !tbaa !6
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 160
  %22 = getelementptr inbounds nuw i8, ptr %2, i64 160
  store <4 x i32> <i32 474, i32 480, i32 486, i32 492>, ptr %21, align 16, !tbaa !6
  store <4 x i32> <i32 1020, i32 1024, i32 1028, i32 1032>, ptr %22, align 16, !tbaa !6
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 176
  %24 = getelementptr inbounds nuw i8, ptr %2, i64 176
  store <4 x i32> <i32 498, i32 504, i32 510, i32 516>, ptr %23, align 16, !tbaa !6
  store <4 x i32> <i32 1036, i32 1040, i32 1044, i32 1048>, ptr %24, align 16, !tbaa !6
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 192
  %26 = getelementptr inbounds nuw i8, ptr %2, i64 192
  store <4 x i32> <i32 553, i32 560, i32 567, i32 574>, ptr %25, align 16, !tbaa !6
  store <4 x i32> <i32 1275, i32 1280, i32 1285, i32 1290>, ptr %26, align 16, !tbaa !6
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 208
  %28 = getelementptr inbounds nuw i8, ptr %2, i64 208
  store <4 x i32> <i32 581, i32 588, i32 595, i32 602>, ptr %27, align 16, !tbaa !6
  store <4 x i32> <i32 1295, i32 1300, i32 1305, i32 1310>, ptr %28, align 16, !tbaa !6
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 224
  %30 = getelementptr inbounds nuw i8, ptr %2, i64 224
  store <4 x i32> <i32 632, i32 640, i32 648, i32 656>, ptr %29, align 16, !tbaa !6
  store <4 x i32> <i32 1530, i32 1536, i32 1542, i32 1548>, ptr %30, align 16, !tbaa !6
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 240
  %32 = getelementptr inbounds nuw i8, ptr %2, i64 240
  store <4 x i32> <i32 664, i32 672, i32 680, i32 688>, ptr %31, align 16, !tbaa !6
  store <4 x i32> <i32 1554, i32 1560, i32 1566, i32 1572>, ptr %32, align 16, !tbaa !6
  %33 = call i32 @my_test(ptr noundef nonnull %1, ptr noundef nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree norecurse nosync nounwind memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }

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
