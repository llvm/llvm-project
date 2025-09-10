; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stdarg-4.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stdarg-4.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@y = dso_local local_unnamed_addr global i64 0, align 8
@x = dso_local local_unnamed_addr global i64 0, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define dso_local void @f1(i32 %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = load ptr, ptr %2, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %5 = load ptr, ptr %4, align 8, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %7 = load ptr, ptr %6, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %9 = load i32, ptr %8, align 8, !tbaa !10
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %11 = load i32, ptr %10, align 4, !tbaa !10
  %12 = icmp sgt i32 %11, -1
  br i1 %12, label %19, label %13

13:                                               ; preds = %1
  %14 = add nsw i32 %11, 16
  %15 = icmp samesign ult i32 %11, -15
  br i1 %15, label %16, label %19

16:                                               ; preds = %13
  %17 = sext i32 %11 to i64
  %18 = getelementptr inbounds i8, ptr %7, i64 %17
  br label %22

19:                                               ; preds = %13, %1
  %20 = phi i32 [ %11, %1 ], [ %14, %13 ]
  %21 = getelementptr inbounds nuw i8, ptr %3, i64 8
  br label %22

22:                                               ; preds = %19, %16
  %23 = phi i32 [ %20, %19 ], [ %14, %16 ]
  %24 = phi ptr [ %21, %19 ], [ %3, %16 ]
  %25 = phi ptr [ %3, %19 ], [ %18, %16 ]
  %26 = load double, ptr %25, align 8, !tbaa !12
  %27 = fptosi double %26 to i64
  store i64 %27, ptr @x, align 8, !tbaa !14
  %28 = icmp slt i32 %9, -7
  %29 = sext i32 %9 to i64
  %30 = getelementptr inbounds i8, ptr %5, i64 %29
  %31 = select i1 %28, i64 0, i64 8
  %32 = getelementptr inbounds nuw i8, ptr %24, i64 %31
  %33 = select i1 %28, ptr %30, ptr %24
  %34 = load i64, ptr %33, align 8, !tbaa !14
  %35 = add nsw i64 %34, %27
  %36 = icmp slt i32 %23, -15
  %37 = sext i32 %23 to i64
  %38 = getelementptr inbounds i8, ptr %7, i64 %37
  %39 = select i1 %36, ptr %38, ptr %32
  %40 = load double, ptr %39, align 8, !tbaa !12
  %41 = sitofp i64 %35 to double
  %42 = fadd double %40, %41
  %43 = fptosi double %42 to i64
  store i64 %43, ptr @x, align 8, !tbaa !14
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define dso_local void @f2(i32 %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = load ptr, ptr %2, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %5 = load ptr, ptr %4, align 8, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %7 = load ptr, ptr %6, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %9 = load i32, ptr %8, align 8, !tbaa !10
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %11 = load i32, ptr %10, align 4, !tbaa !10
  %12 = icmp sgt i32 %9, -1
  br i1 %12, label %16, label %13

13:                                               ; preds = %1
  %14 = add nsw i32 %9, 8
  %15 = icmp samesign ult i32 %9, -7
  br i1 %15, label %21, label %16

16:                                               ; preds = %1, %13
  %17 = phi i32 [ %9, %1 ], [ %14, %13 ]
  %18 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %19 = load i32, ptr %3, align 8, !tbaa !10
  %20 = sext i32 %19 to i64
  store i64 %20, ptr @y, align 8, !tbaa !14
  br label %33

21:                                               ; preds = %13
  %22 = sext i32 %9 to i64
  %23 = getelementptr inbounds i8, ptr %5, i64 %22
  %24 = load i32, ptr %23, align 8, !tbaa !10
  %25 = sext i32 %24 to i64
  store i64 %25, ptr @y, align 8, !tbaa !14
  %26 = icmp eq i32 %9, -8
  br i1 %26, label %33, label %27

27:                                               ; preds = %21
  %28 = add nsw i32 %9, 16
  %29 = icmp samesign ult i32 %14, -7
  br i1 %29, label %30, label %33

30:                                               ; preds = %27
  %31 = sext i32 %14 to i64
  %32 = getelementptr inbounds i8, ptr %5, i64 %31
  br label %38

33:                                               ; preds = %16, %27, %21
  %34 = phi i64 [ %25, %21 ], [ %25, %27 ], [ %20, %16 ]
  %35 = phi ptr [ %3, %21 ], [ %3, %27 ], [ %18, %16 ]
  %36 = phi i32 [ 0, %21 ], [ %28, %27 ], [ %17, %16 ]
  %37 = getelementptr inbounds nuw i8, ptr %35, i64 8
  br label %38

38:                                               ; preds = %33, %30
  %39 = phi i64 [ %34, %33 ], [ %25, %30 ]
  %40 = phi i32 [ %36, %33 ], [ %28, %30 ]
  %41 = phi ptr [ %37, %33 ], [ %3, %30 ]
  %42 = phi ptr [ %35, %33 ], [ %32, %30 ]
  %43 = load i64, ptr %42, align 8, !tbaa !14
  %44 = add nsw i64 %43, %39
  %45 = icmp sgt i32 %11, -1
  br i1 %45, label %49, label %46

46:                                               ; preds = %38
  %47 = add nsw i32 %11, 16
  %48 = icmp samesign ult i32 %11, -15
  br i1 %48, label %56, label %49

49:                                               ; preds = %38, %46
  %50 = phi i32 [ %11, %38 ], [ %47, %46 ]
  %51 = getelementptr inbounds nuw i8, ptr %41, i64 8
  %52 = load double, ptr %41, align 8, !tbaa !12
  %53 = sitofp i64 %44 to double
  %54 = fadd double %52, %53
  %55 = fptosi double %54 to i64
  store i64 %55, ptr @y, align 8, !tbaa !14
  br label %70

56:                                               ; preds = %46
  %57 = sext i32 %11 to i64
  %58 = getelementptr inbounds i8, ptr %7, i64 %57
  %59 = load double, ptr %58, align 8, !tbaa !12
  %60 = sitofp i64 %44 to double
  %61 = fadd double %59, %60
  %62 = fptosi double %61 to i64
  store i64 %62, ptr @y, align 8, !tbaa !14
  %63 = icmp eq i32 %11, -16
  br i1 %63, label %70, label %64

64:                                               ; preds = %56
  %65 = add nsw i32 %11, 32
  %66 = icmp samesign ult i32 %47, -15
  br i1 %66, label %67, label %70

67:                                               ; preds = %64
  %68 = sext i32 %47 to i64
  %69 = getelementptr inbounds i8, ptr %7, i64 %68
  br label %74

70:                                               ; preds = %49, %64, %56
  %71 = phi ptr [ %41, %56 ], [ %41, %64 ], [ %51, %49 ]
  %72 = phi i32 [ 0, %56 ], [ %65, %64 ], [ %50, %49 ]
  %73 = getelementptr inbounds nuw i8, ptr %71, i64 8
  br label %74

74:                                               ; preds = %70, %67
  %75 = phi ptr [ %73, %70 ], [ %41, %67 ]
  %76 = phi i32 [ %72, %70 ], [ %65, %67 ]
  %77 = phi ptr [ %71, %70 ], [ %69, %67 ]
  %78 = load double, ptr %77, align 8, !tbaa !12
  %79 = fptosi double %78 to i64
  store i64 %79, ptr @x, align 8, !tbaa !14
  %80 = icmp slt i32 %40, -7
  %81 = sext i32 %40 to i64
  %82 = getelementptr inbounds i8, ptr %5, i64 %81
  %83 = select i1 %80, i64 0, i64 8
  %84 = getelementptr inbounds nuw i8, ptr %75, i64 %83
  %85 = select i1 %80, ptr %82, ptr %75
  %86 = load i64, ptr %85, align 8, !tbaa !14
  %87 = add nsw i64 %86, %79
  %88 = icmp slt i32 %76, -15
  %89 = sext i32 %76 to i64
  %90 = getelementptr inbounds i8, ptr %7, i64 %89
  %91 = select i1 %88, ptr %90, ptr %84
  %92 = load double, ptr %91, align 8, !tbaa !12
  %93 = sitofp i64 %87 to double
  %94 = fadd double %92, %93
  %95 = fptosi double %94 to i64
  store i64 %95, ptr @x, align 8, !tbaa !14
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i64 @f3h(i32 noundef %0, i64 noundef %1, i64 noundef %2, i64 noundef %3, i64 noundef %4) local_unnamed_addr #3 {
  %6 = sext i32 %0 to i64
  %7 = add nsw i64 %1, %6
  %8 = add nsw i64 %7, %2
  %9 = add nsw i64 %8, %3
  %10 = add nsw i64 %9, %4
  ret i64 %10
}

; Function Attrs: nofree nounwind uwtable
define dso_local i64 @f3(i32 noundef %0, ...) local_unnamed_addr #4 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.va_start.p0(ptr nonnull %2)
  switch i32 %0, label %185 [
    i32 0, label %186
    i32 1, label %3
    i32 2, label %22
    i32 3, label %58
    i32 4, label %112
  ]

3:                                                ; preds = %1
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %5 = load i32, ptr %4, align 8
  %6 = icmp sgt i32 %5, -1
  br i1 %6, label %15, label %7

7:                                                ; preds = %3
  %8 = add nsw i32 %5, 8
  store i32 %8, ptr %4, align 8
  %9 = icmp samesign ult i32 %5, -7
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %12 = load ptr, ptr %11, align 8
  %13 = sext i32 %5 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %18

15:                                               ; preds = %7, %3
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %17, ptr %2, align 8
  br label %18

18:                                               ; preds = %15, %10
  %19 = phi ptr [ %14, %10 ], [ %16, %15 ]
  %20 = load i64, ptr %19, align 8, !tbaa !14
  %21 = add nsw i64 %20, 1
  br label %186

22:                                               ; preds = %1
  %23 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %24 = load i32, ptr %23, align 8
  %25 = icmp sgt i32 %24, -1
  br i1 %25, label %29, label %26

26:                                               ; preds = %22
  %27 = add nsw i32 %24, 8
  store i32 %27, ptr %23, align 8
  %28 = icmp samesign ult i32 %24, -7
  br i1 %28, label %33, label %29

29:                                               ; preds = %22, %26
  %30 = load ptr, ptr %2, align 8
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 8
  store ptr %31, ptr %2, align 8
  %32 = load i64, ptr %30, align 8, !tbaa !14
  br label %48

33:                                               ; preds = %26
  %34 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %35 = load ptr, ptr %34, align 8
  %36 = sext i32 %24 to i64
  %37 = getelementptr inbounds i8, ptr %35, i64 %36
  %38 = load i64, ptr %37, align 8, !tbaa !14
  %39 = icmp sgt i32 %24, -9
  br i1 %39, label %48, label %40

40:                                               ; preds = %33
  %41 = add nsw i32 %24, 16
  store i32 %41, ptr %23, align 8
  %42 = icmp samesign ult i32 %27, -7
  br i1 %42, label %43, label %48

43:                                               ; preds = %40
  %44 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %45 = load ptr, ptr %44, align 8
  %46 = sext i32 %27 to i64
  %47 = getelementptr inbounds i8, ptr %45, i64 %46
  br label %52

48:                                               ; preds = %29, %40, %33
  %49 = phi i64 [ %32, %29 ], [ %38, %40 ], [ %38, %33 ]
  %50 = load ptr, ptr %2, align 8
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 8
  store ptr %51, ptr %2, align 8
  br label %52

52:                                               ; preds = %48, %43
  %53 = phi i64 [ %38, %43 ], [ %49, %48 ]
  %54 = phi ptr [ %47, %43 ], [ %50, %48 ]
  %55 = load i64, ptr %54, align 8, !tbaa !14
  %56 = add nsw i64 %53, 2
  %57 = add nsw i64 %56, %55
  br label %186

58:                                               ; preds = %1
  %59 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %60 = load i32, ptr %59, align 8
  %61 = icmp sgt i32 %60, -1
  br i1 %61, label %65, label %62

62:                                               ; preds = %58
  %63 = add nsw i32 %60, 8
  store i32 %63, ptr %59, align 8
  %64 = icmp samesign ult i32 %60, -7
  br i1 %64, label %69, label %65

65:                                               ; preds = %58, %62
  %66 = load ptr, ptr %2, align 8
  %67 = getelementptr inbounds nuw i8, ptr %66, i64 8
  store ptr %67, ptr %2, align 8
  %68 = load i64, ptr %66, align 8, !tbaa !14
  br label %79

69:                                               ; preds = %62
  %70 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %71 = load ptr, ptr %70, align 8
  %72 = sext i32 %60 to i64
  %73 = getelementptr inbounds i8, ptr %71, i64 %72
  %74 = load i64, ptr %73, align 8, !tbaa !14
  %75 = icmp sgt i32 %60, -9
  br i1 %75, label %79, label %76

76:                                               ; preds = %69
  %77 = add nsw i32 %60, 16
  store i32 %77, ptr %59, align 8
  %78 = icmp samesign ult i32 %63, -7
  br i1 %78, label %84, label %79

79:                                               ; preds = %69, %76, %65
  %80 = phi i64 [ %74, %76 ], [ %74, %69 ], [ %68, %65 ]
  %81 = load ptr, ptr %2, align 8
  %82 = getelementptr inbounds nuw i8, ptr %81, i64 8
  store ptr %82, ptr %2, align 8
  %83 = load i64, ptr %81, align 8, !tbaa !14
  br label %99

84:                                               ; preds = %76
  %85 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %86 = load ptr, ptr %85, align 8
  %87 = sext i32 %63 to i64
  %88 = getelementptr inbounds i8, ptr %86, i64 %87
  %89 = load i64, ptr %88, align 8, !tbaa !14
  %90 = icmp sgt i32 %60, -17
  br i1 %90, label %99, label %91

91:                                               ; preds = %84
  %92 = add nsw i32 %60, 24
  store i32 %92, ptr %59, align 8
  %93 = icmp samesign ult i32 %77, -7
  br i1 %93, label %94, label %99

94:                                               ; preds = %91
  %95 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %96 = load ptr, ptr %95, align 8
  %97 = sext i32 %77 to i64
  %98 = getelementptr inbounds i8, ptr %96, i64 %97
  br label %104

99:                                               ; preds = %79, %91, %84
  %100 = phi i64 [ %83, %79 ], [ %89, %91 ], [ %89, %84 ]
  %101 = phi i64 [ %80, %79 ], [ %74, %91 ], [ %74, %84 ]
  %102 = load ptr, ptr %2, align 8
  %103 = getelementptr inbounds nuw i8, ptr %102, i64 8
  store ptr %103, ptr %2, align 8
  br label %104

104:                                              ; preds = %99, %94
  %105 = phi i64 [ %89, %94 ], [ %100, %99 ]
  %106 = phi i64 [ %74, %94 ], [ %101, %99 ]
  %107 = phi ptr [ %98, %94 ], [ %102, %99 ]
  %108 = load i64, ptr %107, align 8, !tbaa !14
  %109 = add nsw i64 %106, 3
  %110 = add nsw i64 %109, %105
  %111 = add nsw i64 %110, %108
  br label %186

112:                                              ; preds = %1
  %113 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %114 = load i32, ptr %113, align 8
  %115 = icmp sgt i32 %114, -1
  br i1 %115, label %119, label %116

116:                                              ; preds = %112
  %117 = add nsw i32 %114, 8
  store i32 %117, ptr %113, align 8
  %118 = icmp samesign ult i32 %114, -7
  br i1 %118, label %123, label %119

119:                                              ; preds = %112, %116
  %120 = load ptr, ptr %2, align 8
  %121 = getelementptr inbounds nuw i8, ptr %120, i64 8
  store ptr %121, ptr %2, align 8
  %122 = load i64, ptr %120, align 8, !tbaa !14
  br label %133

123:                                              ; preds = %116
  %124 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %125 = load ptr, ptr %124, align 8
  %126 = sext i32 %114 to i64
  %127 = getelementptr inbounds i8, ptr %125, i64 %126
  %128 = load i64, ptr %127, align 8, !tbaa !14
  %129 = icmp sgt i32 %114, -9
  br i1 %129, label %133, label %130

130:                                              ; preds = %123
  %131 = add nsw i32 %114, 16
  store i32 %131, ptr %113, align 8
  %132 = icmp samesign ult i32 %117, -7
  br i1 %132, label %138, label %133

133:                                              ; preds = %123, %130, %119
  %134 = phi i64 [ %128, %130 ], [ %128, %123 ], [ %122, %119 ]
  %135 = load ptr, ptr %2, align 8
  %136 = getelementptr inbounds nuw i8, ptr %135, i64 8
  store ptr %136, ptr %2, align 8
  %137 = load i64, ptr %135, align 8, !tbaa !14
  br label %148

138:                                              ; preds = %130
  %139 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %140 = load ptr, ptr %139, align 8
  %141 = sext i32 %117 to i64
  %142 = getelementptr inbounds i8, ptr %140, i64 %141
  %143 = load i64, ptr %142, align 8, !tbaa !14
  %144 = icmp sgt i32 %114, -17
  br i1 %144, label %148, label %145

145:                                              ; preds = %138
  %146 = add nsw i32 %114, 24
  store i32 %146, ptr %113, align 8
  %147 = icmp samesign ult i32 %131, -7
  br i1 %147, label %154, label %148

148:                                              ; preds = %138, %145, %133
  %149 = phi i64 [ %143, %145 ], [ %143, %138 ], [ %137, %133 ]
  %150 = phi i64 [ %128, %145 ], [ %128, %138 ], [ %134, %133 ]
  %151 = load ptr, ptr %2, align 8
  %152 = getelementptr inbounds nuw i8, ptr %151, i64 8
  store ptr %152, ptr %2, align 8
  %153 = load i64, ptr %151, align 8, !tbaa !14
  br label %169

154:                                              ; preds = %145
  %155 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %156 = load ptr, ptr %155, align 8
  %157 = sext i32 %131 to i64
  %158 = getelementptr inbounds i8, ptr %156, i64 %157
  %159 = load i64, ptr %158, align 8, !tbaa !14
  %160 = icmp sgt i32 %114, -25
  br i1 %160, label %169, label %161

161:                                              ; preds = %154
  %162 = add nsw i32 %114, 32
  store i32 %162, ptr %113, align 8
  %163 = icmp samesign ult i32 %146, -7
  br i1 %163, label %164, label %169

164:                                              ; preds = %161
  %165 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %166 = load ptr, ptr %165, align 8
  %167 = sext i32 %146 to i64
  %168 = getelementptr inbounds i8, ptr %166, i64 %167
  br label %175

169:                                              ; preds = %148, %161, %154
  %170 = phi i64 [ %153, %148 ], [ %159, %161 ], [ %159, %154 ]
  %171 = phi i64 [ %150, %148 ], [ %128, %161 ], [ %128, %154 ]
  %172 = phi i64 [ %149, %148 ], [ %143, %161 ], [ %143, %154 ]
  %173 = load ptr, ptr %2, align 8
  %174 = getelementptr inbounds nuw i8, ptr %173, i64 8
  store ptr %174, ptr %2, align 8
  br label %175

175:                                              ; preds = %169, %164
  %176 = phi i64 [ %159, %164 ], [ %170, %169 ]
  %177 = phi i64 [ %128, %164 ], [ %171, %169 ]
  %178 = phi i64 [ %143, %164 ], [ %172, %169 ]
  %179 = phi ptr [ %168, %164 ], [ %173, %169 ]
  %180 = load i64, ptr %179, align 8, !tbaa !14
  %181 = add nsw i64 %177, 4
  %182 = add nsw i64 %181, %178
  %183 = add nsw i64 %182, %176
  %184 = add nsw i64 %183, %180
  br label %186

185:                                              ; preds = %1
  call void @abort() #7
  unreachable

186:                                              ; preds = %1, %175, %104, %52, %18
  %187 = phi i64 [ %21, %18 ], [ %57, %52 ], [ %111, %104 ], [ %184, %175 ], [ 0, %1 ]
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret i64 %187
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #5

; Function Attrs: nofree nounwind uwtable
define dso_local void @f4(i32 noundef %0, ...) local_unnamed_addr #4 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #6
  call void @llvm.va_start.p0(ptr nonnull %2)
  switch i32 %0, label %64 [
    i32 4, label %3
    i32 5, label %23
  ]

3:                                                ; preds = %1
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %5 = load i32, ptr %4, align 4
  %6 = icmp sgt i32 %5, -1
  br i1 %6, label %15, label %7

7:                                                ; preds = %3
  %8 = add nsw i32 %5, 16
  store i32 %8, ptr %4, align 4
  %9 = icmp samesign ult i32 %5, -15
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %12 = load ptr, ptr %11, align 8
  %13 = sext i32 %5 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %19

15:                                               ; preds = %7, %3
  %16 = phi i32 [ %8, %7 ], [ %5, %3 ]
  %17 = load ptr, ptr %2, align 8
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 8
  store ptr %18, ptr %2, align 8
  br label %19

19:                                               ; preds = %15, %10
  %20 = phi i32 [ %8, %10 ], [ %16, %15 ]
  %21 = phi ptr [ %14, %10 ], [ %17, %15 ]
  %22 = load double, ptr %21, align 8, !tbaa !12
  br label %65

23:                                               ; preds = %1
  %24 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %25 = load i32, ptr %24, align 4
  %26 = icmp sgt i32 %25, -1
  br i1 %26, label %30, label %27

27:                                               ; preds = %23
  %28 = add nsw i32 %25, 16
  store i32 %28, ptr %24, align 4
  %29 = icmp samesign ult i32 %25, -15
  br i1 %29, label %36, label %30

30:                                               ; preds = %23, %27
  %31 = phi i32 [ %28, %27 ], [ %25, %23 ]
  %32 = load ptr, ptr %2, align 8
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 8
  store ptr %33, ptr %2, align 8
  %34 = load double, ptr %32, align 8, !tbaa !12
  %35 = fptosi double %34 to i64
  br label %52

36:                                               ; preds = %27
  %37 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %38 = load ptr, ptr %37, align 8
  %39 = sext i32 %25 to i64
  %40 = getelementptr inbounds i8, ptr %38, i64 %39
  %41 = load double, ptr %40, align 8, !tbaa !12
  %42 = fptosi double %41 to i64
  %43 = icmp sgt i32 %25, -17
  br i1 %43, label %52, label %44

44:                                               ; preds = %36
  %45 = add nsw i32 %25, 32
  store i32 %45, ptr %24, align 4
  %46 = icmp samesign ult i32 %28, -15
  br i1 %46, label %47, label %52

47:                                               ; preds = %44
  %48 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %49 = load ptr, ptr %48, align 8
  %50 = sext i32 %28 to i64
  %51 = getelementptr inbounds i8, ptr %49, i64 %50
  br label %57

52:                                               ; preds = %30, %44, %36
  %53 = phi i64 [ %42, %44 ], [ %42, %36 ], [ %35, %30 ]
  %54 = phi i32 [ %45, %44 ], [ 0, %36 ], [ %31, %30 ]
  %55 = load ptr, ptr %2, align 8
  %56 = getelementptr inbounds nuw i8, ptr %55, i64 8
  store ptr %56, ptr %2, align 8
  br label %57

57:                                               ; preds = %52, %47
  %58 = phi i64 [ %42, %47 ], [ %53, %52 ]
  %59 = phi i32 [ %45, %47 ], [ %54, %52 ]
  %60 = phi ptr [ %51, %47 ], [ %55, %52 ]
  %61 = load double, ptr %60, align 8, !tbaa !12
  %62 = sitofp i64 %58 to double
  %63 = fadd double %61, %62
  br label %65

64:                                               ; preds = %1
  call void @abort() #7
  unreachable

65:                                               ; preds = %57, %19
  %66 = phi i32 [ %59, %57 ], [ %20, %19 ]
  %67 = phi double [ %63, %57 ], [ %22, %19 ]
  %68 = fptosi double %67 to i64
  store i64 %68, ptr @y, align 8, !tbaa !14
  %69 = load ptr, ptr %2, align 8, !tbaa !6
  %70 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %71 = load ptr, ptr %70, align 8, !tbaa !6
  %72 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %73 = load ptr, ptr %72, align 8, !tbaa !6
  %74 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %75 = load i32, ptr %74, align 8, !tbaa !10
  %76 = icmp sgt i32 %66, -1
  br i1 %76, label %83, label %77

77:                                               ; preds = %65
  %78 = add nsw i32 %66, 16
  %79 = icmp samesign ult i32 %66, -15
  br i1 %79, label %80, label %83

80:                                               ; preds = %77
  %81 = sext i32 %66 to i64
  %82 = getelementptr inbounds i8, ptr %73, i64 %81
  br label %86

83:                                               ; preds = %77, %65
  %84 = phi i32 [ %66, %65 ], [ %78, %77 ]
  %85 = getelementptr inbounds nuw i8, ptr %69, i64 8
  br label %86

86:                                               ; preds = %83, %80
  %87 = phi i32 [ %84, %83 ], [ %78, %80 ]
  %88 = phi ptr [ %85, %83 ], [ %69, %80 ]
  %89 = phi ptr [ %69, %83 ], [ %82, %80 ]
  %90 = load double, ptr %89, align 8, !tbaa !12
  %91 = fptosi double %90 to i64
  store i64 %91, ptr @x, align 8, !tbaa !14
  %92 = icmp slt i32 %75, -7
  %93 = sext i32 %75 to i64
  %94 = getelementptr inbounds i8, ptr %71, i64 %93
  %95 = select i1 %92, i64 0, i64 8
  %96 = getelementptr inbounds nuw i8, ptr %88, i64 %95
  %97 = select i1 %92, ptr %94, ptr %88
  %98 = load i64, ptr %97, align 8, !tbaa !14
  %99 = add nsw i64 %98, %91
  %100 = icmp slt i32 %87, -15
  %101 = sext i32 %87 to i64
  %102 = getelementptr inbounds i8, ptr %73, i64 %101
  %103 = select i1 %100, ptr %102, ptr %96
  %104 = load double, ptr %103, align 8, !tbaa !12
  %105 = sitofp i64 %99 to double
  %106 = fadd double %104, %105
  %107 = fptosi double %106 to i64
  store i64 %107, ptr @x, align 8, !tbaa !14
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #6
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 {
  tail call void (i32, ...) @f1(i32 poison, double noundef 1.600000e+01, i64 noundef 128, double noundef 3.200000e+01)
  %1 = load i64, ptr @x, align 8, !tbaa !14
  %2 = icmp eq i64 %1, 176
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

4:                                                ; preds = %0
  tail call void (i32, ...) @f2(i32 poison, i32 noundef 5, i64 noundef 7, double noundef 1.800000e+01, double noundef 1.900000e+01, i64 noundef 17, double noundef 6.400000e+01)
  %5 = load i64, ptr @x, align 8, !tbaa !14
  %6 = icmp ne i64 %5, 100
  %7 = load i64, ptr @y, align 8
  %8 = icmp ne i64 %7, 30
  %9 = select i1 %6, i1 true, i1 %8
  br i1 %9, label %10, label %11

10:                                               ; preds = %4
  tail call void @abort() #7
  unreachable

11:                                               ; preds = %4
  %12 = tail call i64 (i32, ...) @f3(i32 noundef 0)
  %13 = icmp eq i64 %12, 0
  br i1 %13, label %15, label %14

14:                                               ; preds = %11
  tail call void @abort() #7
  unreachable

15:                                               ; preds = %11
  %16 = tail call i64 (i32, ...) @f3(i32 noundef 1, i64 noundef 18)
  %17 = icmp eq i64 %16, 19
  br i1 %17, label %19, label %18

18:                                               ; preds = %15
  tail call void @abort() #7
  unreachable

19:                                               ; preds = %15
  %20 = tail call i64 (i32, ...) @f3(i32 noundef 2, i64 noundef 18, i64 noundef 100)
  %21 = icmp eq i64 %20, 120
  br i1 %21, label %23, label %22

22:                                               ; preds = %19
  tail call void @abort() #7
  unreachable

23:                                               ; preds = %19
  %24 = tail call i64 (i32, ...) @f3(i32 noundef 3, i64 noundef 18, i64 noundef 100, i64 noundef 300)
  %25 = icmp eq i64 %24, 421
  br i1 %25, label %27, label %26

26:                                               ; preds = %23
  tail call void @abort() #7
  unreachable

27:                                               ; preds = %23
  %28 = tail call i64 (i32, ...) @f3(i32 noundef 4, i64 noundef 18, i64 noundef 71, i64 noundef 64, i64 noundef 86)
  %29 = icmp eq i64 %28, 243
  br i1 %29, label %31, label %30

30:                                               ; preds = %27
  tail call void @abort() #7
  unreachable

31:                                               ; preds = %27
  tail call void (i32, ...) @f4(i32 noundef 4, double noundef 6.000000e+00, double noundef 9.000000e+00, i64 noundef 16, double noundef 1.800000e+01)
  %32 = load i64, ptr @x, align 8, !tbaa !14
  %33 = icmp ne i64 %32, 43
  %34 = load i64, ptr @y, align 8
  %35 = icmp ne i64 %34, 6
  %36 = select i1 %33, i1 true, i1 %35
  br i1 %36, label %37, label %38

37:                                               ; preds = %31
  tail call void @abort() #7
  unreachable

38:                                               ; preds = %31
  tail call void (i32, ...) @f4(i32 noundef 5, double noundef 7.000000e+00, double noundef 2.100000e+01, double noundef 1.000000e+00, i64 noundef 17, double noundef 1.260000e+02)
  %39 = load i64, ptr @x, align 8, !tbaa !14
  %40 = icmp ne i64 %39, 144
  %41 = load i64, ptr @y, align 8
  %42 = icmp ne i64 %41, 28
  %43 = select i1 %40, i1 true, i1 %42
  br i1 %43, label %44, label %45

44:                                               ; preds = %38
  tail call void @abort() #7
  unreachable

45:                                               ; preds = %38
  ret i32 0
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nounwind }
attributes #7 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"any pointer", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"double", !8, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"long", !8, i64 0}
