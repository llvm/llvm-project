; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/va-arg-24.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/va-arg-24.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@errors = internal unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [9 x i8] c"varargs0\00", align 1
@.str.1 = private unnamed_addr constant [29 x i8] c" %s: n[%d] = %d expected %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [9 x i8] c"varargs1\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"varargs2\00", align 1
@.str.4 = private unnamed_addr constant [9 x i8] c"varargs3\00", align 1
@.str.5 = private unnamed_addr constant [9 x i8] c"varargs4\00", align 1
@.str.6 = private unnamed_addr constant [9 x i8] c"varargs5\00", align 1
@.str.7 = private unnamed_addr constant [9 x i8] c"varargs6\00", align 1
@.str.8 = private unnamed_addr constant [9 x i8] c"varargs7\00", align 1
@.str.9 = private unnamed_addr constant [9 x i8] c"varargs8\00", align 1
@.str.10 = private unnamed_addr constant [9 x i8] c"varargs9\00", align 1

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void (i32, ...) @varargs0(i32 poison, i32 noundef 1, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10)
  tail call void (i32, i32, ...) @varargs1(i32 poison, i32 poison, i32 noundef 2, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10)
  tail call void (i32, i32, i32, ...) @varargs2(i32 poison, i32 poison, i32 poison, i32 noundef 3, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10)
  tail call void (i32, i32, i32, i32, ...) @varargs3(i32 poison, i32 poison, i32 poison, i32 poison, i32 noundef 4, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10)
  tail call void (i32, i32, i32, i32, i32, ...) @varargs4(i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10)
  tail call void (i32, i32, i32, i32, i32, i32, ...) @varargs5(i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 noundef 6, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10)
  tail call void (i32, i32, i32, i32, i32, i32, i32, ...) @varargs6(i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 noundef 7, i32 noundef 8, i32 noundef 9, i32 noundef 10)
  tail call void (i32, i32, i32, i32, i32, i32, i32, i32, ...) @varargs7(i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 noundef 8, i32 noundef 9, i32 noundef 10)
  tail call void (i32, i32, i32, i32, i32, i32, i32, i32, i32, ...) @varargs8(i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 noundef 9, i32 noundef 10)
  tail call void (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, ...) @varargs9(i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 noundef 10)
  %1 = load i32, ptr @errors, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

4:                                                ; preds = %0
  tail call void @exit(i32 noundef 0) #8
  unreachable
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs0(i32 %0, ...) unnamed_addr #1 {
  %2 = alloca %struct.__va_list, align 8
  %3 = alloca [11 x i32], align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #9
  call void @llvm.va_start.p0(ptr nonnull %2)
  store i32 0, ptr %3, align 4, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %5 = load i32, ptr %4, align 8
  %6 = load ptr, ptr %2, align 8
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %8 = load ptr, ptr %7, align 8
  %9 = icmp sgt i32 %5, -1
  br i1 %9, label %13, label %10

10:                                               ; preds = %1
  %11 = add nsw i32 %5, 8
  store i32 %11, ptr %4, align 8
  %12 = icmp samesign ult i32 %5, -7
  br i1 %12, label %17, label %13

13:                                               ; preds = %1, %10
  %14 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store ptr %14, ptr %2, align 8
  %15 = load i32, ptr %6, align 8, !tbaa !6
  %16 = getelementptr inbounds nuw i8, ptr %3, i64 4
  store i32 %15, ptr %16, align 4, !tbaa !6
  br label %26

17:                                               ; preds = %10
  %18 = sext i32 %5 to i64
  %19 = getelementptr inbounds i8, ptr %8, i64 %18
  %20 = load i32, ptr %19, align 8, !tbaa !6
  %21 = getelementptr inbounds nuw i8, ptr %3, i64 4
  store i32 %20, ptr %21, align 4, !tbaa !6
  %22 = icmp sgt i32 %5, -9
  br i1 %22, label %26, label %23

23:                                               ; preds = %17
  %24 = add nsw i32 %5, 16
  store i32 %24, ptr %4, align 8
  %25 = icmp samesign ult i32 %11, -7
  br i1 %25, label %31, label %26

26:                                               ; preds = %17, %23, %13
  %27 = phi ptr [ %6, %23 ], [ %6, %17 ], [ %14, %13 ]
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 8
  store ptr %28, ptr %2, align 8
  %29 = load i32, ptr %27, align 8, !tbaa !6
  %30 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i32 %29, ptr %30, align 4, !tbaa !6
  br label %40

31:                                               ; preds = %23
  %32 = sext i32 %11 to i64
  %33 = getelementptr inbounds i8, ptr %8, i64 %32
  %34 = load i32, ptr %33, align 8, !tbaa !6
  %35 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i32 %34, ptr %35, align 4, !tbaa !6
  %36 = icmp sgt i32 %5, -17
  br i1 %36, label %40, label %37

37:                                               ; preds = %31
  %38 = add nsw i32 %5, 24
  store i32 %38, ptr %4, align 8
  %39 = icmp samesign ult i32 %24, -7
  br i1 %39, label %45, label %40

40:                                               ; preds = %31, %37, %26
  %41 = phi ptr [ %6, %37 ], [ %6, %31 ], [ %28, %26 ]
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 8
  store ptr %42, ptr %2, align 8
  %43 = load i32, ptr %41, align 8, !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %3, i64 12
  store i32 %43, ptr %44, align 4, !tbaa !6
  br label %54

45:                                               ; preds = %37
  %46 = sext i32 %24 to i64
  %47 = getelementptr inbounds i8, ptr %8, i64 %46
  %48 = load i32, ptr %47, align 8, !tbaa !6
  %49 = getelementptr inbounds nuw i8, ptr %3, i64 12
  store i32 %48, ptr %49, align 4, !tbaa !6
  %50 = icmp sgt i32 %5, -25
  br i1 %50, label %54, label %51

51:                                               ; preds = %45
  %52 = add nsw i32 %5, 32
  store i32 %52, ptr %4, align 8
  %53 = icmp samesign ult i32 %38, -7
  br i1 %53, label %59, label %54

54:                                               ; preds = %45, %51, %40
  %55 = phi ptr [ %6, %51 ], [ %6, %45 ], [ %42, %40 ]
  %56 = getelementptr inbounds nuw i8, ptr %55, i64 8
  store ptr %56, ptr %2, align 8
  %57 = load i32, ptr %55, align 8, !tbaa !6
  %58 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i32 %57, ptr %58, align 4, !tbaa !6
  br label %68

59:                                               ; preds = %51
  %60 = sext i32 %38 to i64
  %61 = getelementptr inbounds i8, ptr %8, i64 %60
  %62 = load i32, ptr %61, align 8, !tbaa !6
  %63 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i32 %62, ptr %63, align 4, !tbaa !6
  %64 = icmp sgt i32 %5, -33
  br i1 %64, label %68, label %65

65:                                               ; preds = %59
  %66 = add nsw i32 %5, 40
  store i32 %66, ptr %4, align 8
  %67 = icmp samesign ult i32 %52, -7
  br i1 %67, label %73, label %68

68:                                               ; preds = %59, %65, %54
  %69 = phi ptr [ %6, %65 ], [ %6, %59 ], [ %56, %54 ]
  %70 = getelementptr inbounds nuw i8, ptr %69, i64 8
  store ptr %70, ptr %2, align 8
  %71 = load i32, ptr %69, align 8, !tbaa !6
  %72 = getelementptr inbounds nuw i8, ptr %3, i64 20
  store i32 %71, ptr %72, align 4, !tbaa !6
  br label %82

73:                                               ; preds = %65
  %74 = sext i32 %52 to i64
  %75 = getelementptr inbounds i8, ptr %8, i64 %74
  %76 = load i32, ptr %75, align 8, !tbaa !6
  %77 = getelementptr inbounds nuw i8, ptr %3, i64 20
  store i32 %76, ptr %77, align 4, !tbaa !6
  %78 = icmp sgt i32 %5, -41
  br i1 %78, label %82, label %79

79:                                               ; preds = %73
  %80 = add nsw i32 %5, 48
  store i32 %80, ptr %4, align 8
  %81 = icmp samesign ult i32 %66, -7
  br i1 %81, label %87, label %82

82:                                               ; preds = %73, %79, %68
  %83 = phi ptr [ %6, %79 ], [ %6, %73 ], [ %70, %68 ]
  %84 = getelementptr inbounds nuw i8, ptr %83, i64 8
  store ptr %84, ptr %2, align 8
  %85 = load i32, ptr %83, align 8, !tbaa !6
  %86 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store i32 %85, ptr %86, align 4, !tbaa !6
  br label %96

87:                                               ; preds = %79
  %88 = sext i32 %66 to i64
  %89 = getelementptr inbounds i8, ptr %8, i64 %88
  %90 = load i32, ptr %89, align 8, !tbaa !6
  %91 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store i32 %90, ptr %91, align 4, !tbaa !6
  %92 = icmp sgt i32 %5, -49
  br i1 %92, label %96, label %93

93:                                               ; preds = %87
  %94 = add nsw i32 %5, 56
  store i32 %94, ptr %4, align 8
  %95 = icmp samesign ult i32 %80, -7
  br i1 %95, label %101, label %96

96:                                               ; preds = %87, %93, %82
  %97 = phi ptr [ %6, %93 ], [ %6, %87 ], [ %84, %82 ]
  %98 = getelementptr inbounds nuw i8, ptr %97, i64 8
  store ptr %98, ptr %2, align 8
  %99 = load i32, ptr %97, align 8, !tbaa !6
  %100 = getelementptr inbounds nuw i8, ptr %3, i64 28
  store i32 %99, ptr %100, align 4, !tbaa !6
  br label %110

101:                                              ; preds = %93
  %102 = sext i32 %80 to i64
  %103 = getelementptr inbounds i8, ptr %8, i64 %102
  %104 = load i32, ptr %103, align 8, !tbaa !6
  %105 = getelementptr inbounds nuw i8, ptr %3, i64 28
  store i32 %104, ptr %105, align 4, !tbaa !6
  %106 = icmp sgt i32 %5, -57
  br i1 %106, label %110, label %107

107:                                              ; preds = %101
  %108 = add nsw i32 %5, 64
  store i32 %108, ptr %4, align 8
  %109 = icmp samesign ult i32 %94, -7
  br i1 %109, label %115, label %110

110:                                              ; preds = %101, %107, %96
  %111 = phi ptr [ %6, %107 ], [ %6, %101 ], [ %98, %96 ]
  %112 = getelementptr inbounds nuw i8, ptr %111, i64 8
  store ptr %112, ptr %2, align 8
  %113 = load i32, ptr %111, align 8, !tbaa !6
  %114 = getelementptr inbounds nuw i8, ptr %3, i64 32
  store i32 %113, ptr %114, align 4, !tbaa !6
  br label %124

115:                                              ; preds = %107
  %116 = sext i32 %94 to i64
  %117 = getelementptr inbounds i8, ptr %8, i64 %116
  %118 = load i32, ptr %117, align 8, !tbaa !6
  %119 = getelementptr inbounds nuw i8, ptr %3, i64 32
  store i32 %118, ptr %119, align 4, !tbaa !6
  %120 = icmp sgt i32 %5, -65
  br i1 %120, label %124, label %121

121:                                              ; preds = %115
  %122 = add nsw i32 %5, 72
  store i32 %122, ptr %4, align 8
  %123 = icmp samesign ult i32 %108, -7
  br i1 %123, label %129, label %124

124:                                              ; preds = %115, %121, %110
  %125 = phi ptr [ %6, %121 ], [ %6, %115 ], [ %112, %110 ]
  %126 = getelementptr inbounds nuw i8, ptr %125, i64 8
  store ptr %126, ptr %2, align 8
  %127 = load i32, ptr %125, align 8, !tbaa !6
  %128 = getelementptr inbounds nuw i8, ptr %3, i64 36
  store i32 %127, ptr %128, align 4, !tbaa !6
  br label %141

129:                                              ; preds = %121
  %130 = sext i32 %108 to i64
  %131 = getelementptr inbounds i8, ptr %8, i64 %130
  %132 = load i32, ptr %131, align 8, !tbaa !6
  %133 = getelementptr inbounds nuw i8, ptr %3, i64 36
  store i32 %132, ptr %133, align 4, !tbaa !6
  %134 = icmp sgt i32 %5, -73
  br i1 %134, label %141, label %135

135:                                              ; preds = %129
  %136 = add nsw i32 %5, 80
  store i32 %136, ptr %4, align 8
  %137 = icmp samesign ult i32 %122, -7
  br i1 %137, label %138, label %141

138:                                              ; preds = %135
  %139 = sext i32 %122 to i64
  %140 = getelementptr inbounds i8, ptr %8, i64 %139
  br label %144

141:                                              ; preds = %124, %135, %129
  %142 = phi ptr [ %126, %124 ], [ %6, %135 ], [ %6, %129 ]
  %143 = getelementptr inbounds nuw i8, ptr %142, i64 8
  store ptr %143, ptr %2, align 8
  br label %144

144:                                              ; preds = %141, %138
  %145 = phi ptr [ %140, %138 ], [ %142, %141 ]
  %146 = load i32, ptr %145, align 8, !tbaa !6
  %147 = getelementptr inbounds nuw i8, ptr %3, i64 40
  store i32 %146, ptr %147, align 4, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %2)
  call fastcc void @verify(ptr noundef nonnull @.str, ptr noundef %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs1(i32 %0, i32 %1, ...) unnamed_addr #1 {
  %3 = alloca %struct.__va_list, align 8
  %4 = alloca [11 x i32], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #9
  call void @llvm.va_start.p0(ptr nonnull %3)
  store <2 x i32> <i32 0, i32 1>, ptr %4, align 8, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %6 = load i32, ptr %5, align 8
  %7 = load ptr, ptr %3, align 8
  %8 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %9 = load ptr, ptr %8, align 8
  %10 = icmp sgt i32 %6, -1
  br i1 %10, label %14, label %11

11:                                               ; preds = %2
  %12 = add nsw i32 %6, 8
  store i32 %12, ptr %5, align 8
  %13 = icmp samesign ult i32 %6, -7
  br i1 %13, label %18, label %14

14:                                               ; preds = %2, %11
  %15 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store ptr %15, ptr %3, align 8
  %16 = load i32, ptr %7, align 8, !tbaa !6
  %17 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i32 %16, ptr %17, align 8, !tbaa !6
  br label %27

18:                                               ; preds = %11
  %19 = sext i32 %6 to i64
  %20 = getelementptr inbounds i8, ptr %9, i64 %19
  %21 = load i32, ptr %20, align 8, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i32 %21, ptr %22, align 8, !tbaa !6
  %23 = icmp sgt i32 %6, -9
  br i1 %23, label %27, label %24

24:                                               ; preds = %18
  %25 = add nsw i32 %6, 16
  store i32 %25, ptr %5, align 8
  %26 = icmp samesign ult i32 %12, -7
  br i1 %26, label %32, label %27

27:                                               ; preds = %18, %24, %14
  %28 = phi ptr [ %7, %24 ], [ %7, %18 ], [ %15, %14 ]
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 8
  store ptr %29, ptr %3, align 8
  %30 = load i32, ptr %28, align 8, !tbaa !6
  %31 = getelementptr inbounds nuw i8, ptr %4, i64 12
  store i32 %30, ptr %31, align 4, !tbaa !6
  br label %41

32:                                               ; preds = %24
  %33 = sext i32 %12 to i64
  %34 = getelementptr inbounds i8, ptr %9, i64 %33
  %35 = load i32, ptr %34, align 8, !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %4, i64 12
  store i32 %35, ptr %36, align 4, !tbaa !6
  %37 = icmp sgt i32 %6, -17
  br i1 %37, label %41, label %38

38:                                               ; preds = %32
  %39 = add nsw i32 %6, 24
  store i32 %39, ptr %5, align 8
  %40 = icmp samesign ult i32 %25, -7
  br i1 %40, label %46, label %41

41:                                               ; preds = %32, %38, %27
  %42 = phi ptr [ %7, %38 ], [ %7, %32 ], [ %29, %27 ]
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 8
  store ptr %43, ptr %3, align 8
  %44 = load i32, ptr %42, align 8, !tbaa !6
  %45 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 %44, ptr %45, align 8, !tbaa !6
  br label %55

46:                                               ; preds = %38
  %47 = sext i32 %25 to i64
  %48 = getelementptr inbounds i8, ptr %9, i64 %47
  %49 = load i32, ptr %48, align 8, !tbaa !6
  %50 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 %49, ptr %50, align 8, !tbaa !6
  %51 = icmp sgt i32 %6, -25
  br i1 %51, label %55, label %52

52:                                               ; preds = %46
  %53 = add nsw i32 %6, 32
  store i32 %53, ptr %5, align 8
  %54 = icmp samesign ult i32 %39, -7
  br i1 %54, label %60, label %55

55:                                               ; preds = %46, %52, %41
  %56 = phi ptr [ %7, %52 ], [ %7, %46 ], [ %43, %41 ]
  %57 = getelementptr inbounds nuw i8, ptr %56, i64 8
  store ptr %57, ptr %3, align 8
  %58 = load i32, ptr %56, align 8, !tbaa !6
  %59 = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i32 %58, ptr %59, align 4, !tbaa !6
  br label %69

60:                                               ; preds = %52
  %61 = sext i32 %39 to i64
  %62 = getelementptr inbounds i8, ptr %9, i64 %61
  %63 = load i32, ptr %62, align 8, !tbaa !6
  %64 = getelementptr inbounds nuw i8, ptr %4, i64 20
  store i32 %63, ptr %64, align 4, !tbaa !6
  %65 = icmp sgt i32 %6, -33
  br i1 %65, label %69, label %66

66:                                               ; preds = %60
  %67 = add nsw i32 %6, 40
  store i32 %67, ptr %5, align 8
  %68 = icmp samesign ult i32 %53, -7
  br i1 %68, label %74, label %69

69:                                               ; preds = %60, %66, %55
  %70 = phi ptr [ %7, %66 ], [ %7, %60 ], [ %57, %55 ]
  %71 = getelementptr inbounds nuw i8, ptr %70, i64 8
  store ptr %71, ptr %3, align 8
  %72 = load i32, ptr %70, align 8, !tbaa !6
  %73 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store i32 %72, ptr %73, align 8, !tbaa !6
  br label %83

74:                                               ; preds = %66
  %75 = sext i32 %53 to i64
  %76 = getelementptr inbounds i8, ptr %9, i64 %75
  %77 = load i32, ptr %76, align 8, !tbaa !6
  %78 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store i32 %77, ptr %78, align 8, !tbaa !6
  %79 = icmp sgt i32 %6, -41
  br i1 %79, label %83, label %80

80:                                               ; preds = %74
  %81 = add nsw i32 %6, 48
  store i32 %81, ptr %5, align 8
  %82 = icmp samesign ult i32 %67, -7
  br i1 %82, label %88, label %83

83:                                               ; preds = %74, %80, %69
  %84 = phi ptr [ %7, %80 ], [ %7, %74 ], [ %71, %69 ]
  %85 = getelementptr inbounds nuw i8, ptr %84, i64 8
  store ptr %85, ptr %3, align 8
  %86 = load i32, ptr %84, align 8, !tbaa !6
  %87 = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i32 %86, ptr %87, align 4, !tbaa !6
  br label %97

88:                                               ; preds = %80
  %89 = sext i32 %67 to i64
  %90 = getelementptr inbounds i8, ptr %9, i64 %89
  %91 = load i32, ptr %90, align 8, !tbaa !6
  %92 = getelementptr inbounds nuw i8, ptr %4, i64 28
  store i32 %91, ptr %92, align 4, !tbaa !6
  %93 = icmp sgt i32 %6, -49
  br i1 %93, label %97, label %94

94:                                               ; preds = %88
  %95 = add nsw i32 %6, 56
  store i32 %95, ptr %5, align 8
  %96 = icmp samesign ult i32 %81, -7
  br i1 %96, label %102, label %97

97:                                               ; preds = %88, %94, %83
  %98 = phi ptr [ %7, %94 ], [ %7, %88 ], [ %85, %83 ]
  %99 = getelementptr inbounds nuw i8, ptr %98, i64 8
  store ptr %99, ptr %3, align 8
  %100 = load i32, ptr %98, align 8, !tbaa !6
  %101 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store i32 %100, ptr %101, align 8, !tbaa !6
  br label %111

102:                                              ; preds = %94
  %103 = sext i32 %81 to i64
  %104 = getelementptr inbounds i8, ptr %9, i64 %103
  %105 = load i32, ptr %104, align 8, !tbaa !6
  %106 = getelementptr inbounds nuw i8, ptr %4, i64 32
  store i32 %105, ptr %106, align 8, !tbaa !6
  %107 = icmp sgt i32 %6, -57
  br i1 %107, label %111, label %108

108:                                              ; preds = %102
  %109 = add nsw i32 %6, 64
  store i32 %109, ptr %5, align 8
  %110 = icmp samesign ult i32 %95, -7
  br i1 %110, label %116, label %111

111:                                              ; preds = %102, %108, %97
  %112 = phi ptr [ %7, %108 ], [ %7, %102 ], [ %99, %97 ]
  %113 = getelementptr inbounds nuw i8, ptr %112, i64 8
  store ptr %113, ptr %3, align 8
  %114 = load i32, ptr %112, align 8, !tbaa !6
  %115 = getelementptr inbounds nuw i8, ptr %4, i64 36
  store i32 %114, ptr %115, align 4, !tbaa !6
  br label %128

116:                                              ; preds = %108
  %117 = sext i32 %95 to i64
  %118 = getelementptr inbounds i8, ptr %9, i64 %117
  %119 = load i32, ptr %118, align 8, !tbaa !6
  %120 = getelementptr inbounds nuw i8, ptr %4, i64 36
  store i32 %119, ptr %120, align 4, !tbaa !6
  %121 = icmp sgt i32 %6, -65
  br i1 %121, label %128, label %122

122:                                              ; preds = %116
  %123 = add nsw i32 %6, 72
  store i32 %123, ptr %5, align 8
  %124 = icmp samesign ult i32 %109, -7
  br i1 %124, label %125, label %128

125:                                              ; preds = %122
  %126 = sext i32 %109 to i64
  %127 = getelementptr inbounds i8, ptr %9, i64 %126
  br label %131

128:                                              ; preds = %111, %122, %116
  %129 = phi ptr [ %113, %111 ], [ %7, %122 ], [ %7, %116 ]
  %130 = getelementptr inbounds nuw i8, ptr %129, i64 8
  store ptr %130, ptr %3, align 8
  br label %131

131:                                              ; preds = %128, %125
  %132 = phi ptr [ %127, %125 ], [ %129, %128 ]
  %133 = load i32, ptr %132, align 8, !tbaa !6
  %134 = getelementptr inbounds nuw i8, ptr %4, i64 40
  store i32 %133, ptr %134, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %3)
  call fastcc void @verify(ptr noundef nonnull @.str.2, ptr noundef %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs2(i32 %0, i32 %1, i32 %2, ...) unnamed_addr #1 {
  %4 = alloca %struct.__va_list, align 8
  %5 = alloca [11 x i32], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #9
  call void @llvm.va_start.p0(ptr nonnull %4)
  store <2 x i32> <i32 0, i32 1>, ptr %5, align 8, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i32 2, ptr %6, align 8, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %8 = load i32, ptr %7, align 8
  %9 = load ptr, ptr %4, align 8
  %10 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %11 = load ptr, ptr %10, align 8
  %12 = icmp sgt i32 %8, -1
  br i1 %12, label %16, label %13

13:                                               ; preds = %3
  %14 = add nsw i32 %8, 8
  store i32 %14, ptr %7, align 8
  %15 = icmp samesign ult i32 %8, -7
  br i1 %15, label %20, label %16

16:                                               ; preds = %3, %13
  %17 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store ptr %17, ptr %4, align 8
  %18 = load i32, ptr %9, align 8, !tbaa !6
  %19 = getelementptr inbounds nuw i8, ptr %5, i64 12
  store i32 %18, ptr %19, align 4, !tbaa !6
  br label %29

20:                                               ; preds = %13
  %21 = sext i32 %8 to i64
  %22 = getelementptr inbounds i8, ptr %11, i64 %21
  %23 = load i32, ptr %22, align 8, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %5, i64 12
  store i32 %23, ptr %24, align 4, !tbaa !6
  %25 = icmp sgt i32 %8, -9
  br i1 %25, label %29, label %26

26:                                               ; preds = %20
  %27 = add nsw i32 %8, 16
  store i32 %27, ptr %7, align 8
  %28 = icmp samesign ult i32 %14, -7
  br i1 %28, label %34, label %29

29:                                               ; preds = %20, %26, %16
  %30 = phi ptr [ %9, %26 ], [ %9, %20 ], [ %17, %16 ]
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 8
  store ptr %31, ptr %4, align 8
  %32 = load i32, ptr %30, align 8, !tbaa !6
  %33 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store i32 %32, ptr %33, align 8, !tbaa !6
  br label %43

34:                                               ; preds = %26
  %35 = sext i32 %14 to i64
  %36 = getelementptr inbounds i8, ptr %11, i64 %35
  %37 = load i32, ptr %36, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store i32 %37, ptr %38, align 8, !tbaa !6
  %39 = icmp sgt i32 %8, -17
  br i1 %39, label %43, label %40

40:                                               ; preds = %34
  %41 = add nsw i32 %8, 24
  store i32 %41, ptr %7, align 8
  %42 = icmp samesign ult i32 %27, -7
  br i1 %42, label %48, label %43

43:                                               ; preds = %34, %40, %29
  %44 = phi ptr [ %9, %40 ], [ %9, %34 ], [ %31, %29 ]
  %45 = getelementptr inbounds nuw i8, ptr %44, i64 8
  store ptr %45, ptr %4, align 8
  %46 = load i32, ptr %44, align 8, !tbaa !6
  %47 = getelementptr inbounds nuw i8, ptr %5, i64 20
  store i32 %46, ptr %47, align 4, !tbaa !6
  br label %57

48:                                               ; preds = %40
  %49 = sext i32 %27 to i64
  %50 = getelementptr inbounds i8, ptr %11, i64 %49
  %51 = load i32, ptr %50, align 8, !tbaa !6
  %52 = getelementptr inbounds nuw i8, ptr %5, i64 20
  store i32 %51, ptr %52, align 4, !tbaa !6
  %53 = icmp sgt i32 %8, -25
  br i1 %53, label %57, label %54

54:                                               ; preds = %48
  %55 = add nsw i32 %8, 32
  store i32 %55, ptr %7, align 8
  %56 = icmp samesign ult i32 %41, -7
  br i1 %56, label %62, label %57

57:                                               ; preds = %48, %54, %43
  %58 = phi ptr [ %9, %54 ], [ %9, %48 ], [ %45, %43 ]
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 8
  store ptr %59, ptr %4, align 8
  %60 = load i32, ptr %58, align 8, !tbaa !6
  %61 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i32 %60, ptr %61, align 8, !tbaa !6
  br label %71

62:                                               ; preds = %54
  %63 = sext i32 %41 to i64
  %64 = getelementptr inbounds i8, ptr %11, i64 %63
  %65 = load i32, ptr %64, align 8, !tbaa !6
  %66 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store i32 %65, ptr %66, align 8, !tbaa !6
  %67 = icmp sgt i32 %8, -33
  br i1 %67, label %71, label %68

68:                                               ; preds = %62
  %69 = add nsw i32 %8, 40
  store i32 %69, ptr %7, align 8
  %70 = icmp samesign ult i32 %55, -7
  br i1 %70, label %76, label %71

71:                                               ; preds = %62, %68, %57
  %72 = phi ptr [ %9, %68 ], [ %9, %62 ], [ %59, %57 ]
  %73 = getelementptr inbounds nuw i8, ptr %72, i64 8
  store ptr %73, ptr %4, align 8
  %74 = load i32, ptr %72, align 8, !tbaa !6
  %75 = getelementptr inbounds nuw i8, ptr %5, i64 28
  store i32 %74, ptr %75, align 4, !tbaa !6
  br label %85

76:                                               ; preds = %68
  %77 = sext i32 %55 to i64
  %78 = getelementptr inbounds i8, ptr %11, i64 %77
  %79 = load i32, ptr %78, align 8, !tbaa !6
  %80 = getelementptr inbounds nuw i8, ptr %5, i64 28
  store i32 %79, ptr %80, align 4, !tbaa !6
  %81 = icmp sgt i32 %8, -41
  br i1 %81, label %85, label %82

82:                                               ; preds = %76
  %83 = add nsw i32 %8, 48
  store i32 %83, ptr %7, align 8
  %84 = icmp samesign ult i32 %69, -7
  br i1 %84, label %90, label %85

85:                                               ; preds = %76, %82, %71
  %86 = phi ptr [ %9, %82 ], [ %9, %76 ], [ %73, %71 ]
  %87 = getelementptr inbounds nuw i8, ptr %86, i64 8
  store ptr %87, ptr %4, align 8
  %88 = load i32, ptr %86, align 8, !tbaa !6
  %89 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store i32 %88, ptr %89, align 8, !tbaa !6
  br label %99

90:                                               ; preds = %82
  %91 = sext i32 %69 to i64
  %92 = getelementptr inbounds i8, ptr %11, i64 %91
  %93 = load i32, ptr %92, align 8, !tbaa !6
  %94 = getelementptr inbounds nuw i8, ptr %5, i64 32
  store i32 %93, ptr %94, align 8, !tbaa !6
  %95 = icmp sgt i32 %8, -49
  br i1 %95, label %99, label %96

96:                                               ; preds = %90
  %97 = add nsw i32 %8, 56
  store i32 %97, ptr %7, align 8
  %98 = icmp samesign ult i32 %83, -7
  br i1 %98, label %104, label %99

99:                                               ; preds = %90, %96, %85
  %100 = phi ptr [ %9, %96 ], [ %9, %90 ], [ %87, %85 ]
  %101 = getelementptr inbounds nuw i8, ptr %100, i64 8
  store ptr %101, ptr %4, align 8
  %102 = load i32, ptr %100, align 8, !tbaa !6
  %103 = getelementptr inbounds nuw i8, ptr %5, i64 36
  store i32 %102, ptr %103, align 4, !tbaa !6
  br label %116

104:                                              ; preds = %96
  %105 = sext i32 %83 to i64
  %106 = getelementptr inbounds i8, ptr %11, i64 %105
  %107 = load i32, ptr %106, align 8, !tbaa !6
  %108 = getelementptr inbounds nuw i8, ptr %5, i64 36
  store i32 %107, ptr %108, align 4, !tbaa !6
  %109 = icmp sgt i32 %8, -57
  br i1 %109, label %116, label %110

110:                                              ; preds = %104
  %111 = add nsw i32 %8, 64
  store i32 %111, ptr %7, align 8
  %112 = icmp samesign ult i32 %97, -7
  br i1 %112, label %113, label %116

113:                                              ; preds = %110
  %114 = sext i32 %97 to i64
  %115 = getelementptr inbounds i8, ptr %11, i64 %114
  br label %119

116:                                              ; preds = %99, %110, %104
  %117 = phi ptr [ %101, %99 ], [ %9, %110 ], [ %9, %104 ]
  %118 = getelementptr inbounds nuw i8, ptr %117, i64 8
  store ptr %118, ptr %4, align 8
  br label %119

119:                                              ; preds = %116, %113
  %120 = phi ptr [ %115, %113 ], [ %117, %116 ]
  %121 = load i32, ptr %120, align 8, !tbaa !6
  %122 = getelementptr inbounds nuw i8, ptr %5, i64 40
  store i32 %121, ptr %122, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %4)
  call fastcc void @verify(ptr noundef nonnull @.str.3, ptr noundef %5)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs3(i32 %0, i32 %1, i32 %2, i32 %3, ...) unnamed_addr #1 {
  %5 = alloca %struct.__va_list, align 8
  %6 = alloca [11 x i32], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #9
  call void @llvm.va_start.p0(ptr nonnull %5)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr %6, align 16, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %5, i64 24
  %8 = load i32, ptr %7, align 8
  %9 = load ptr, ptr %5, align 8
  %10 = getelementptr inbounds nuw i8, ptr %5, i64 8
  %11 = load ptr, ptr %10, align 8
  %12 = icmp sgt i32 %8, -1
  br i1 %12, label %16, label %13

13:                                               ; preds = %4
  %14 = add nsw i32 %8, 8
  store i32 %14, ptr %7, align 8
  %15 = icmp samesign ult i32 %8, -7
  br i1 %15, label %20, label %16

16:                                               ; preds = %4, %13
  %17 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store ptr %17, ptr %5, align 8
  %18 = load i32, ptr %9, align 8, !tbaa !6
  %19 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store i32 %18, ptr %19, align 16, !tbaa !6
  br label %29

20:                                               ; preds = %13
  %21 = sext i32 %8 to i64
  %22 = getelementptr inbounds i8, ptr %11, i64 %21
  %23 = load i32, ptr %22, align 8, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store i32 %23, ptr %24, align 16, !tbaa !6
  %25 = icmp sgt i32 %8, -9
  br i1 %25, label %29, label %26

26:                                               ; preds = %20
  %27 = add nsw i32 %8, 16
  store i32 %27, ptr %7, align 8
  %28 = icmp samesign ult i32 %14, -7
  br i1 %28, label %34, label %29

29:                                               ; preds = %20, %26, %16
  %30 = phi ptr [ %9, %26 ], [ %9, %20 ], [ %17, %16 ]
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 8
  store ptr %31, ptr %5, align 8
  %32 = load i32, ptr %30, align 8, !tbaa !6
  %33 = getelementptr inbounds nuw i8, ptr %6, i64 20
  store i32 %32, ptr %33, align 4, !tbaa !6
  br label %43

34:                                               ; preds = %26
  %35 = sext i32 %14 to i64
  %36 = getelementptr inbounds i8, ptr %11, i64 %35
  %37 = load i32, ptr %36, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %6, i64 20
  store i32 %37, ptr %38, align 4, !tbaa !6
  %39 = icmp sgt i32 %8, -17
  br i1 %39, label %43, label %40

40:                                               ; preds = %34
  %41 = add nsw i32 %8, 24
  store i32 %41, ptr %7, align 8
  %42 = icmp samesign ult i32 %27, -7
  br i1 %42, label %48, label %43

43:                                               ; preds = %34, %40, %29
  %44 = phi ptr [ %9, %40 ], [ %9, %34 ], [ %31, %29 ]
  %45 = getelementptr inbounds nuw i8, ptr %44, i64 8
  store ptr %45, ptr %5, align 8
  %46 = load i32, ptr %44, align 8, !tbaa !6
  %47 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store i32 %46, ptr %47, align 8, !tbaa !6
  br label %57

48:                                               ; preds = %40
  %49 = sext i32 %27 to i64
  %50 = getelementptr inbounds i8, ptr %11, i64 %49
  %51 = load i32, ptr %50, align 8, !tbaa !6
  %52 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store i32 %51, ptr %52, align 8, !tbaa !6
  %53 = icmp sgt i32 %8, -25
  br i1 %53, label %57, label %54

54:                                               ; preds = %48
  %55 = add nsw i32 %8, 32
  store i32 %55, ptr %7, align 8
  %56 = icmp samesign ult i32 %41, -7
  br i1 %56, label %62, label %57

57:                                               ; preds = %48, %54, %43
  %58 = phi ptr [ %9, %54 ], [ %9, %48 ], [ %45, %43 ]
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 8
  store ptr %59, ptr %5, align 8
  %60 = load i32, ptr %58, align 8, !tbaa !6
  %61 = getelementptr inbounds nuw i8, ptr %6, i64 28
  store i32 %60, ptr %61, align 4, !tbaa !6
  br label %71

62:                                               ; preds = %54
  %63 = sext i32 %41 to i64
  %64 = getelementptr inbounds i8, ptr %11, i64 %63
  %65 = load i32, ptr %64, align 8, !tbaa !6
  %66 = getelementptr inbounds nuw i8, ptr %6, i64 28
  store i32 %65, ptr %66, align 4, !tbaa !6
  %67 = icmp sgt i32 %8, -33
  br i1 %67, label %71, label %68

68:                                               ; preds = %62
  %69 = add nsw i32 %8, 40
  store i32 %69, ptr %7, align 8
  %70 = icmp samesign ult i32 %55, -7
  br i1 %70, label %76, label %71

71:                                               ; preds = %62, %68, %57
  %72 = phi ptr [ %9, %68 ], [ %9, %62 ], [ %59, %57 ]
  %73 = getelementptr inbounds nuw i8, ptr %72, i64 8
  store ptr %73, ptr %5, align 8
  %74 = load i32, ptr %72, align 8, !tbaa !6
  %75 = getelementptr inbounds nuw i8, ptr %6, i64 32
  store i32 %74, ptr %75, align 16, !tbaa !6
  br label %85

76:                                               ; preds = %68
  %77 = sext i32 %55 to i64
  %78 = getelementptr inbounds i8, ptr %11, i64 %77
  %79 = load i32, ptr %78, align 8, !tbaa !6
  %80 = getelementptr inbounds nuw i8, ptr %6, i64 32
  store i32 %79, ptr %80, align 16, !tbaa !6
  %81 = icmp sgt i32 %8, -41
  br i1 %81, label %85, label %82

82:                                               ; preds = %76
  %83 = add nsw i32 %8, 48
  store i32 %83, ptr %7, align 8
  %84 = icmp samesign ult i32 %69, -7
  br i1 %84, label %90, label %85

85:                                               ; preds = %76, %82, %71
  %86 = phi ptr [ %9, %82 ], [ %9, %76 ], [ %73, %71 ]
  %87 = getelementptr inbounds nuw i8, ptr %86, i64 8
  store ptr %87, ptr %5, align 8
  %88 = load i32, ptr %86, align 8, !tbaa !6
  %89 = getelementptr inbounds nuw i8, ptr %6, i64 36
  store i32 %88, ptr %89, align 4, !tbaa !6
  br label %102

90:                                               ; preds = %82
  %91 = sext i32 %69 to i64
  %92 = getelementptr inbounds i8, ptr %11, i64 %91
  %93 = load i32, ptr %92, align 8, !tbaa !6
  %94 = getelementptr inbounds nuw i8, ptr %6, i64 36
  store i32 %93, ptr %94, align 4, !tbaa !6
  %95 = icmp sgt i32 %8, -49
  br i1 %95, label %102, label %96

96:                                               ; preds = %90
  %97 = add nsw i32 %8, 56
  store i32 %97, ptr %7, align 8
  %98 = icmp samesign ult i32 %83, -7
  br i1 %98, label %99, label %102

99:                                               ; preds = %96
  %100 = sext i32 %83 to i64
  %101 = getelementptr inbounds i8, ptr %11, i64 %100
  br label %105

102:                                              ; preds = %85, %96, %90
  %103 = phi ptr [ %87, %85 ], [ %9, %96 ], [ %9, %90 ]
  %104 = getelementptr inbounds nuw i8, ptr %103, i64 8
  store ptr %104, ptr %5, align 8
  br label %105

105:                                              ; preds = %102, %99
  %106 = phi ptr [ %101, %99 ], [ %103, %102 ]
  %107 = load i32, ptr %106, align 8, !tbaa !6
  %108 = getelementptr inbounds nuw i8, ptr %6, i64 40
  store i32 %107, ptr %108, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %5)
  call fastcc void @verify(ptr noundef nonnull @.str.4, ptr noundef %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs4(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, ...) unnamed_addr #1 {
  %6 = alloca %struct.__va_list, align 8
  %7 = alloca [11 x i32], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #9
  call void @llvm.va_start.p0(ptr nonnull %6)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr %7, align 16, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store i32 4, ptr %8, align 16, !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %6, i64 24
  %10 = load i32, ptr %9, align 8
  %11 = load ptr, ptr %6, align 8
  %12 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %13 = load ptr, ptr %12, align 8
  %14 = icmp sgt i32 %10, -1
  br i1 %14, label %18, label %15

15:                                               ; preds = %5
  %16 = add nsw i32 %10, 8
  store i32 %16, ptr %9, align 8
  %17 = icmp samesign ult i32 %10, -7
  br i1 %17, label %22, label %18

18:                                               ; preds = %5, %15
  %19 = getelementptr inbounds nuw i8, ptr %11, i64 8
  store ptr %19, ptr %6, align 8
  %20 = load i32, ptr %11, align 8, !tbaa !6
  %21 = getelementptr inbounds nuw i8, ptr %7, i64 20
  store i32 %20, ptr %21, align 4, !tbaa !6
  br label %31

22:                                               ; preds = %15
  %23 = sext i32 %10 to i64
  %24 = getelementptr inbounds i8, ptr %13, i64 %23
  %25 = load i32, ptr %24, align 8, !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %7, i64 20
  store i32 %25, ptr %26, align 4, !tbaa !6
  %27 = icmp sgt i32 %10, -9
  br i1 %27, label %31, label %28

28:                                               ; preds = %22
  %29 = add nsw i32 %10, 16
  store i32 %29, ptr %9, align 8
  %30 = icmp samesign ult i32 %16, -7
  br i1 %30, label %36, label %31

31:                                               ; preds = %22, %28, %18
  %32 = phi ptr [ %11, %28 ], [ %11, %22 ], [ %19, %18 ]
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 8
  store ptr %33, ptr %6, align 8
  %34 = load i32, ptr %32, align 8, !tbaa !6
  %35 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store i32 %34, ptr %35, align 8, !tbaa !6
  br label %45

36:                                               ; preds = %28
  %37 = sext i32 %16 to i64
  %38 = getelementptr inbounds i8, ptr %13, i64 %37
  %39 = load i32, ptr %38, align 8, !tbaa !6
  %40 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store i32 %39, ptr %40, align 8, !tbaa !6
  %41 = icmp sgt i32 %10, -17
  br i1 %41, label %45, label %42

42:                                               ; preds = %36
  %43 = add nsw i32 %10, 24
  store i32 %43, ptr %9, align 8
  %44 = icmp samesign ult i32 %29, -7
  br i1 %44, label %50, label %45

45:                                               ; preds = %36, %42, %31
  %46 = phi ptr [ %11, %42 ], [ %11, %36 ], [ %33, %31 ]
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 8
  store ptr %47, ptr %6, align 8
  %48 = load i32, ptr %46, align 8, !tbaa !6
  %49 = getelementptr inbounds nuw i8, ptr %7, i64 28
  store i32 %48, ptr %49, align 4, !tbaa !6
  br label %59

50:                                               ; preds = %42
  %51 = sext i32 %29 to i64
  %52 = getelementptr inbounds i8, ptr %13, i64 %51
  %53 = load i32, ptr %52, align 8, !tbaa !6
  %54 = getelementptr inbounds nuw i8, ptr %7, i64 28
  store i32 %53, ptr %54, align 4, !tbaa !6
  %55 = icmp sgt i32 %10, -25
  br i1 %55, label %59, label %56

56:                                               ; preds = %50
  %57 = add nsw i32 %10, 32
  store i32 %57, ptr %9, align 8
  %58 = icmp samesign ult i32 %43, -7
  br i1 %58, label %64, label %59

59:                                               ; preds = %50, %56, %45
  %60 = phi ptr [ %11, %56 ], [ %11, %50 ], [ %47, %45 ]
  %61 = getelementptr inbounds nuw i8, ptr %60, i64 8
  store ptr %61, ptr %6, align 8
  %62 = load i32, ptr %60, align 8, !tbaa !6
  %63 = getelementptr inbounds nuw i8, ptr %7, i64 32
  store i32 %62, ptr %63, align 16, !tbaa !6
  br label %73

64:                                               ; preds = %56
  %65 = sext i32 %43 to i64
  %66 = getelementptr inbounds i8, ptr %13, i64 %65
  %67 = load i32, ptr %66, align 8, !tbaa !6
  %68 = getelementptr inbounds nuw i8, ptr %7, i64 32
  store i32 %67, ptr %68, align 16, !tbaa !6
  %69 = icmp sgt i32 %10, -33
  br i1 %69, label %73, label %70

70:                                               ; preds = %64
  %71 = add nsw i32 %10, 40
  store i32 %71, ptr %9, align 8
  %72 = icmp samesign ult i32 %57, -7
  br i1 %72, label %78, label %73

73:                                               ; preds = %64, %70, %59
  %74 = phi ptr [ %11, %70 ], [ %11, %64 ], [ %61, %59 ]
  %75 = getelementptr inbounds nuw i8, ptr %74, i64 8
  store ptr %75, ptr %6, align 8
  %76 = load i32, ptr %74, align 8, !tbaa !6
  %77 = getelementptr inbounds nuw i8, ptr %7, i64 36
  store i32 %76, ptr %77, align 4, !tbaa !6
  br label %90

78:                                               ; preds = %70
  %79 = sext i32 %57 to i64
  %80 = getelementptr inbounds i8, ptr %13, i64 %79
  %81 = load i32, ptr %80, align 8, !tbaa !6
  %82 = getelementptr inbounds nuw i8, ptr %7, i64 36
  store i32 %81, ptr %82, align 4, !tbaa !6
  %83 = icmp sgt i32 %10, -41
  br i1 %83, label %90, label %84

84:                                               ; preds = %78
  %85 = add nsw i32 %10, 48
  store i32 %85, ptr %9, align 8
  %86 = icmp samesign ult i32 %71, -7
  br i1 %86, label %87, label %90

87:                                               ; preds = %84
  %88 = sext i32 %71 to i64
  %89 = getelementptr inbounds i8, ptr %13, i64 %88
  br label %93

90:                                               ; preds = %73, %84, %78
  %91 = phi ptr [ %75, %73 ], [ %11, %84 ], [ %11, %78 ]
  %92 = getelementptr inbounds nuw i8, ptr %91, i64 8
  store ptr %92, ptr %6, align 8
  br label %93

93:                                               ; preds = %90, %87
  %94 = phi ptr [ %89, %87 ], [ %91, %90 ]
  %95 = load i32, ptr %94, align 8, !tbaa !6
  %96 = getelementptr inbounds nuw i8, ptr %7, i64 40
  store i32 %95, ptr %96, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %6)
  call fastcc void @verify(ptr noundef nonnull @.str.5, ptr noundef %7)
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs5(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, ...) unnamed_addr #1 {
  %7 = alloca %struct.__va_list, align 8
  %8 = alloca [11 x i32], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #9
  call void @llvm.va_start.p0(ptr nonnull %7)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr %8, align 16, !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store <2 x i32> <i32 4, i32 5>, ptr %9, align 16, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %7, i64 24
  %11 = load i32, ptr %10, align 8
  %12 = load ptr, ptr %7, align 8
  %13 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %14 = load ptr, ptr %13, align 8
  %15 = icmp sgt i32 %11, -1
  br i1 %15, label %19, label %16

16:                                               ; preds = %6
  %17 = add nsw i32 %11, 8
  store i32 %17, ptr %10, align 8
  %18 = icmp samesign ult i32 %11, -7
  br i1 %18, label %23, label %19

19:                                               ; preds = %6, %16
  %20 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store ptr %20, ptr %7, align 8
  %21 = load i32, ptr %12, align 8, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %8, i64 24
  store i32 %21, ptr %22, align 8, !tbaa !6
  br label %32

23:                                               ; preds = %16
  %24 = sext i32 %11 to i64
  %25 = getelementptr inbounds i8, ptr %14, i64 %24
  %26 = load i32, ptr %25, align 8, !tbaa !6
  %27 = getelementptr inbounds nuw i8, ptr %8, i64 24
  store i32 %26, ptr %27, align 8, !tbaa !6
  %28 = icmp sgt i32 %11, -9
  br i1 %28, label %32, label %29

29:                                               ; preds = %23
  %30 = add nsw i32 %11, 16
  store i32 %30, ptr %10, align 8
  %31 = icmp samesign ult i32 %17, -7
  br i1 %31, label %37, label %32

32:                                               ; preds = %23, %29, %19
  %33 = phi ptr [ %12, %29 ], [ %12, %23 ], [ %20, %19 ]
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 8
  store ptr %34, ptr %7, align 8
  %35 = load i32, ptr %33, align 8, !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %8, i64 28
  store i32 %35, ptr %36, align 4, !tbaa !6
  br label %46

37:                                               ; preds = %29
  %38 = sext i32 %17 to i64
  %39 = getelementptr inbounds i8, ptr %14, i64 %38
  %40 = load i32, ptr %39, align 8, !tbaa !6
  %41 = getelementptr inbounds nuw i8, ptr %8, i64 28
  store i32 %40, ptr %41, align 4, !tbaa !6
  %42 = icmp sgt i32 %11, -17
  br i1 %42, label %46, label %43

43:                                               ; preds = %37
  %44 = add nsw i32 %11, 24
  store i32 %44, ptr %10, align 8
  %45 = icmp samesign ult i32 %30, -7
  br i1 %45, label %51, label %46

46:                                               ; preds = %37, %43, %32
  %47 = phi ptr [ %12, %43 ], [ %12, %37 ], [ %34, %32 ]
  %48 = getelementptr inbounds nuw i8, ptr %47, i64 8
  store ptr %48, ptr %7, align 8
  %49 = load i32, ptr %47, align 8, !tbaa !6
  %50 = getelementptr inbounds nuw i8, ptr %8, i64 32
  store i32 %49, ptr %50, align 16, !tbaa !6
  br label %60

51:                                               ; preds = %43
  %52 = sext i32 %30 to i64
  %53 = getelementptr inbounds i8, ptr %14, i64 %52
  %54 = load i32, ptr %53, align 8, !tbaa !6
  %55 = getelementptr inbounds nuw i8, ptr %8, i64 32
  store i32 %54, ptr %55, align 16, !tbaa !6
  %56 = icmp sgt i32 %11, -25
  br i1 %56, label %60, label %57

57:                                               ; preds = %51
  %58 = add nsw i32 %11, 32
  store i32 %58, ptr %10, align 8
  %59 = icmp samesign ult i32 %44, -7
  br i1 %59, label %65, label %60

60:                                               ; preds = %51, %57, %46
  %61 = phi ptr [ %12, %57 ], [ %12, %51 ], [ %48, %46 ]
  %62 = getelementptr inbounds nuw i8, ptr %61, i64 8
  store ptr %62, ptr %7, align 8
  %63 = load i32, ptr %61, align 8, !tbaa !6
  %64 = getelementptr inbounds nuw i8, ptr %8, i64 36
  store i32 %63, ptr %64, align 4, !tbaa !6
  br label %77

65:                                               ; preds = %57
  %66 = sext i32 %44 to i64
  %67 = getelementptr inbounds i8, ptr %14, i64 %66
  %68 = load i32, ptr %67, align 8, !tbaa !6
  %69 = getelementptr inbounds nuw i8, ptr %8, i64 36
  store i32 %68, ptr %69, align 4, !tbaa !6
  %70 = icmp sgt i32 %11, -33
  br i1 %70, label %77, label %71

71:                                               ; preds = %65
  %72 = add nsw i32 %11, 40
  store i32 %72, ptr %10, align 8
  %73 = icmp samesign ult i32 %58, -7
  br i1 %73, label %74, label %77

74:                                               ; preds = %71
  %75 = sext i32 %58 to i64
  %76 = getelementptr inbounds i8, ptr %14, i64 %75
  br label %80

77:                                               ; preds = %60, %71, %65
  %78 = phi ptr [ %62, %60 ], [ %12, %71 ], [ %12, %65 ]
  %79 = getelementptr inbounds nuw i8, ptr %78, i64 8
  store ptr %79, ptr %7, align 8
  br label %80

80:                                               ; preds = %77, %74
  %81 = phi ptr [ %76, %74 ], [ %78, %77 ]
  %82 = load i32, ptr %81, align 8, !tbaa !6
  %83 = getelementptr inbounds nuw i8, ptr %8, i64 40
  store i32 %82, ptr %83, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %7)
  call fastcc void @verify(ptr noundef nonnull @.str.6, ptr noundef %8)
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs6(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, ...) unnamed_addr #1 {
  %8 = alloca %struct.__va_list, align 8
  %9 = alloca [11 x i32], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #9
  call void @llvm.va_start.p0(ptr nonnull %8)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr %9, align 16, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store <2 x i32> <i32 4, i32 5>, ptr %10, align 16, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %9, i64 24
  store i32 6, ptr %11, align 8, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %8, i64 24
  %13 = load i32, ptr %12, align 8
  %14 = load ptr, ptr %8, align 8
  %15 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %16 = load ptr, ptr %15, align 8
  %17 = icmp sgt i32 %13, -1
  br i1 %17, label %21, label %18

18:                                               ; preds = %7
  %19 = add nsw i32 %13, 8
  store i32 %19, ptr %12, align 8
  %20 = icmp samesign ult i32 %13, -7
  br i1 %20, label %25, label %21

21:                                               ; preds = %7, %18
  %22 = getelementptr inbounds nuw i8, ptr %14, i64 8
  store ptr %22, ptr %8, align 8
  %23 = load i32, ptr %14, align 8, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %9, i64 28
  store i32 %23, ptr %24, align 4, !tbaa !6
  br label %34

25:                                               ; preds = %18
  %26 = sext i32 %13 to i64
  %27 = getelementptr inbounds i8, ptr %16, i64 %26
  %28 = load i32, ptr %27, align 8, !tbaa !6
  %29 = getelementptr inbounds nuw i8, ptr %9, i64 28
  store i32 %28, ptr %29, align 4, !tbaa !6
  %30 = icmp sgt i32 %13, -9
  br i1 %30, label %34, label %31

31:                                               ; preds = %25
  %32 = add nsw i32 %13, 16
  store i32 %32, ptr %12, align 8
  %33 = icmp samesign ult i32 %19, -7
  br i1 %33, label %39, label %34

34:                                               ; preds = %25, %31, %21
  %35 = phi ptr [ %14, %31 ], [ %14, %25 ], [ %22, %21 ]
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 8
  store ptr %36, ptr %8, align 8
  %37 = load i32, ptr %35, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %9, i64 32
  store i32 %37, ptr %38, align 16, !tbaa !6
  br label %48

39:                                               ; preds = %31
  %40 = sext i32 %19 to i64
  %41 = getelementptr inbounds i8, ptr %16, i64 %40
  %42 = load i32, ptr %41, align 8, !tbaa !6
  %43 = getelementptr inbounds nuw i8, ptr %9, i64 32
  store i32 %42, ptr %43, align 16, !tbaa !6
  %44 = icmp sgt i32 %13, -17
  br i1 %44, label %48, label %45

45:                                               ; preds = %39
  %46 = add nsw i32 %13, 24
  store i32 %46, ptr %12, align 8
  %47 = icmp samesign ult i32 %32, -7
  br i1 %47, label %53, label %48

48:                                               ; preds = %39, %45, %34
  %49 = phi ptr [ %14, %45 ], [ %14, %39 ], [ %36, %34 ]
  %50 = getelementptr inbounds nuw i8, ptr %49, i64 8
  store ptr %50, ptr %8, align 8
  %51 = load i32, ptr %49, align 8, !tbaa !6
  %52 = getelementptr inbounds nuw i8, ptr %9, i64 36
  store i32 %51, ptr %52, align 4, !tbaa !6
  br label %65

53:                                               ; preds = %45
  %54 = sext i32 %32 to i64
  %55 = getelementptr inbounds i8, ptr %16, i64 %54
  %56 = load i32, ptr %55, align 8, !tbaa !6
  %57 = getelementptr inbounds nuw i8, ptr %9, i64 36
  store i32 %56, ptr %57, align 4, !tbaa !6
  %58 = icmp sgt i32 %13, -25
  br i1 %58, label %65, label %59

59:                                               ; preds = %53
  %60 = add nsw i32 %13, 32
  store i32 %60, ptr %12, align 8
  %61 = icmp samesign ult i32 %46, -7
  br i1 %61, label %62, label %65

62:                                               ; preds = %59
  %63 = sext i32 %46 to i64
  %64 = getelementptr inbounds i8, ptr %16, i64 %63
  br label %68

65:                                               ; preds = %48, %59, %53
  %66 = phi ptr [ %50, %48 ], [ %14, %59 ], [ %14, %53 ]
  %67 = getelementptr inbounds nuw i8, ptr %66, i64 8
  store ptr %67, ptr %8, align 8
  br label %68

68:                                               ; preds = %65, %62
  %69 = phi ptr [ %64, %62 ], [ %66, %65 ]
  %70 = load i32, ptr %69, align 8, !tbaa !6
  %71 = getelementptr inbounds nuw i8, ptr %9, i64 40
  store i32 %70, ptr %71, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %8)
  call fastcc void @verify(ptr noundef nonnull @.str.7, ptr noundef %9)
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs7(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, ...) unnamed_addr #1 {
  %9 = alloca %struct.__va_list, align 8
  %10 = alloca [11 x i32], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #9
  call void @llvm.va_start.p0(ptr nonnull %9)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr %10, align 16, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 16
  store <4 x i32> <i32 4, i32 5, i32 6, i32 7>, ptr %11, align 16, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %9, i64 24
  %13 = load i32, ptr %12, align 8
  %14 = load ptr, ptr %9, align 8
  %15 = getelementptr inbounds nuw i8, ptr %9, i64 8
  %16 = load ptr, ptr %15, align 8
  %17 = icmp sgt i32 %13, -1
  br i1 %17, label %21, label %18

18:                                               ; preds = %8
  %19 = add nsw i32 %13, 8
  store i32 %19, ptr %12, align 8
  %20 = icmp samesign ult i32 %13, -7
  br i1 %20, label %25, label %21

21:                                               ; preds = %8, %18
  %22 = getelementptr inbounds nuw i8, ptr %14, i64 8
  store ptr %22, ptr %9, align 8
  %23 = load i32, ptr %14, align 8, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %10, i64 32
  store i32 %23, ptr %24, align 16, !tbaa !6
  br label %34

25:                                               ; preds = %18
  %26 = sext i32 %13 to i64
  %27 = getelementptr inbounds i8, ptr %16, i64 %26
  %28 = load i32, ptr %27, align 8, !tbaa !6
  %29 = getelementptr inbounds nuw i8, ptr %10, i64 32
  store i32 %28, ptr %29, align 16, !tbaa !6
  %30 = icmp sgt i32 %13, -9
  br i1 %30, label %34, label %31

31:                                               ; preds = %25
  %32 = add nsw i32 %13, 16
  store i32 %32, ptr %12, align 8
  %33 = icmp samesign ult i32 %19, -7
  br i1 %33, label %39, label %34

34:                                               ; preds = %25, %31, %21
  %35 = phi ptr [ %14, %31 ], [ %14, %25 ], [ %22, %21 ]
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 8
  store ptr %36, ptr %9, align 8
  %37 = load i32, ptr %35, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %10, i64 36
  store i32 %37, ptr %38, align 4, !tbaa !6
  br label %51

39:                                               ; preds = %31
  %40 = sext i32 %19 to i64
  %41 = getelementptr inbounds i8, ptr %16, i64 %40
  %42 = load i32, ptr %41, align 8, !tbaa !6
  %43 = getelementptr inbounds nuw i8, ptr %10, i64 36
  store i32 %42, ptr %43, align 4, !tbaa !6
  %44 = icmp sgt i32 %13, -17
  br i1 %44, label %51, label %45

45:                                               ; preds = %39
  %46 = add nsw i32 %13, 24
  store i32 %46, ptr %12, align 8
  %47 = icmp samesign ult i32 %32, -7
  br i1 %47, label %48, label %51

48:                                               ; preds = %45
  %49 = sext i32 %32 to i64
  %50 = getelementptr inbounds i8, ptr %16, i64 %49
  br label %54

51:                                               ; preds = %34, %45, %39
  %52 = phi ptr [ %36, %34 ], [ %14, %45 ], [ %14, %39 ]
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 8
  store ptr %53, ptr %9, align 8
  br label %54

54:                                               ; preds = %51, %48
  %55 = phi ptr [ %50, %48 ], [ %52, %51 ]
  %56 = load i32, ptr %55, align 8, !tbaa !6
  %57 = getelementptr inbounds nuw i8, ptr %10, i64 40
  store i32 %56, ptr %57, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %9)
  call fastcc void @verify(ptr noundef nonnull @.str.8, ptr noundef %10)
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs8(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, ...) unnamed_addr #1 {
  %10 = alloca %struct.__va_list, align 8
  %11 = alloca [11 x i32], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #9
  call void @llvm.va_start.p0(ptr nonnull %10)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr %11, align 16, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store <4 x i32> <i32 4, i32 5, i32 6, i32 7>, ptr %12, align 16, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %11, i64 32
  store i32 8, ptr %13, align 16, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %10, i64 24
  %15 = load i32, ptr %14, align 8
  %16 = load ptr, ptr %10, align 8
  %17 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %18 = load ptr, ptr %17, align 8
  %19 = icmp sgt i32 %15, -1
  br i1 %19, label %23, label %20

20:                                               ; preds = %9
  %21 = add nsw i32 %15, 8
  store i32 %21, ptr %14, align 8
  %22 = icmp samesign ult i32 %15, -7
  br i1 %22, label %27, label %23

23:                                               ; preds = %9, %20
  %24 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %24, ptr %10, align 8
  %25 = load i32, ptr %16, align 8, !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %11, i64 36
  store i32 %25, ptr %26, align 4, !tbaa !6
  br label %39

27:                                               ; preds = %20
  %28 = sext i32 %15 to i64
  %29 = getelementptr inbounds i8, ptr %18, i64 %28
  %30 = load i32, ptr %29, align 8, !tbaa !6
  %31 = getelementptr inbounds nuw i8, ptr %11, i64 36
  store i32 %30, ptr %31, align 4, !tbaa !6
  %32 = icmp sgt i32 %15, -9
  br i1 %32, label %39, label %33

33:                                               ; preds = %27
  %34 = add nsw i32 %15, 16
  store i32 %34, ptr %14, align 8
  %35 = icmp samesign ult i32 %21, -7
  br i1 %35, label %36, label %39

36:                                               ; preds = %33
  %37 = sext i32 %21 to i64
  %38 = getelementptr inbounds i8, ptr %18, i64 %37
  br label %42

39:                                               ; preds = %23, %33, %27
  %40 = phi ptr [ %24, %23 ], [ %16, %33 ], [ %16, %27 ]
  %41 = getelementptr inbounds nuw i8, ptr %40, i64 8
  store ptr %41, ptr %10, align 8
  br label %42

42:                                               ; preds = %39, %36
  %43 = phi ptr [ %38, %36 ], [ %40, %39 ]
  %44 = load i32, ptr %43, align 8, !tbaa !6
  %45 = getelementptr inbounds nuw i8, ptr %11, i64 40
  store i32 %44, ptr %45, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %10)
  call fastcc void @verify(ptr noundef nonnull @.str.9, ptr noundef %11)
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define internal void @varargs9(i32 %0, i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 %6, i32 %7, i32 %8, i32 %9, ...) unnamed_addr #1 {
  %11 = alloca %struct.__va_list, align 8
  %12 = alloca [11 x i32], align 16
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #9
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #9
  call void @llvm.va_start.p0(ptr nonnull %11)
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr %12, align 16, !tbaa !6
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 16
  store <4 x i32> <i32 4, i32 5, i32 6, i32 7>, ptr %13, align 16, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 32
  store <2 x i32> <i32 8, i32 9>, ptr %14, align 16, !tbaa !6
  %15 = load ptr, ptr %11, align 8
  %16 = getelementptr inbounds nuw i8, ptr %11, i64 24
  %17 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %18 = load ptr, ptr %17, align 8
  %19 = load i32, ptr %16, align 8
  %20 = icmp sgt i32 %19, -1
  br i1 %20, label %27, label %21

21:                                               ; preds = %10
  %22 = add nsw i32 %19, 8
  store i32 %22, ptr %16, align 8
  %23 = icmp samesign ult i32 %19, -7
  br i1 %23, label %24, label %27

24:                                               ; preds = %21
  %25 = sext i32 %19 to i64
  %26 = getelementptr inbounds i8, ptr %18, i64 %25
  br label %29

27:                                               ; preds = %21, %10
  %28 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store ptr %28, ptr %11, align 8
  br label %29

29:                                               ; preds = %27, %24
  %30 = phi ptr [ %26, %24 ], [ %15, %27 ]
  %31 = getelementptr inbounds nuw i8, ptr %12, i64 40
  %32 = load i32, ptr %30, align 8, !tbaa !6
  store i32 %32, ptr %31, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %11)
  call fastcc void @verify(ptr noundef nonnull @.str.10, ptr noundef %12)
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #9
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #9
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #5

; Function Attrs: nofree nounwind uwtable
define internal fastcc void @verify(ptr noundef %0, ptr noundef nonnull readonly captures(none) %1) unnamed_addr #1 {
  %3 = load i32, ptr %1, align 4, !tbaa !6
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %9, label %5

5:                                                ; preds = %2
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 0, i32 noundef %3, i32 noundef 0)
  %7 = load i32, ptr @errors, align 4, !tbaa !6
  %8 = add nsw i32 %7, 1
  store i32 %8, ptr @errors, align 4, !tbaa !6
  br label %9

9:                                                ; preds = %2, %5
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %11 = load i32, ptr %10, align 4, !tbaa !6
  %12 = icmp eq i32 %11, 1
  br i1 %12, label %17, label %13

13:                                               ; preds = %9
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 1, i32 noundef %11, i32 noundef 1)
  %15 = load i32, ptr @errors, align 4, !tbaa !6
  %16 = add nsw i32 %15, 1
  store i32 %16, ptr @errors, align 4, !tbaa !6
  br label %17

17:                                               ; preds = %13, %9
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %19 = load i32, ptr %18, align 4, !tbaa !6
  %20 = icmp eq i32 %19, 2
  br i1 %20, label %25, label %21

21:                                               ; preds = %17
  %22 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 2, i32 noundef %19, i32 noundef 2)
  %23 = load i32, ptr @errors, align 4, !tbaa !6
  %24 = add nsw i32 %23, 1
  store i32 %24, ptr @errors, align 4, !tbaa !6
  br label %25

25:                                               ; preds = %21, %17
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %27 = load i32, ptr %26, align 4, !tbaa !6
  %28 = icmp eq i32 %27, 3
  br i1 %28, label %33, label %29

29:                                               ; preds = %25
  %30 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 3, i32 noundef %27, i32 noundef 3)
  %31 = load i32, ptr @errors, align 4, !tbaa !6
  %32 = add nsw i32 %31, 1
  store i32 %32, ptr @errors, align 4, !tbaa !6
  br label %33

33:                                               ; preds = %29, %25
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %35 = load i32, ptr %34, align 4, !tbaa !6
  %36 = icmp eq i32 %35, 4
  br i1 %36, label %41, label %37

37:                                               ; preds = %33
  %38 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 4, i32 noundef %35, i32 noundef 4)
  %39 = load i32, ptr @errors, align 4, !tbaa !6
  %40 = add nsw i32 %39, 1
  store i32 %40, ptr @errors, align 4, !tbaa !6
  br label %41

41:                                               ; preds = %37, %33
  %42 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %43 = load i32, ptr %42, align 4, !tbaa !6
  %44 = icmp eq i32 %43, 5
  br i1 %44, label %49, label %45

45:                                               ; preds = %41
  %46 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 5, i32 noundef %43, i32 noundef 5)
  %47 = load i32, ptr @errors, align 4, !tbaa !6
  %48 = add nsw i32 %47, 1
  store i32 %48, ptr @errors, align 4, !tbaa !6
  br label %49

49:                                               ; preds = %45, %41
  %50 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %51 = load i32, ptr %50, align 4, !tbaa !6
  %52 = icmp eq i32 %51, 6
  br i1 %52, label %57, label %53

53:                                               ; preds = %49
  %54 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 6, i32 noundef %51, i32 noundef 6)
  %55 = load i32, ptr @errors, align 4, !tbaa !6
  %56 = add nsw i32 %55, 1
  store i32 %56, ptr @errors, align 4, !tbaa !6
  br label %57

57:                                               ; preds = %53, %49
  %58 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %59 = load i32, ptr %58, align 4, !tbaa !6
  %60 = icmp eq i32 %59, 7
  br i1 %60, label %65, label %61

61:                                               ; preds = %57
  %62 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 7, i32 noundef %59, i32 noundef 7)
  %63 = load i32, ptr @errors, align 4, !tbaa !6
  %64 = add nsw i32 %63, 1
  store i32 %64, ptr @errors, align 4, !tbaa !6
  br label %65

65:                                               ; preds = %61, %57
  %66 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %67 = load i32, ptr %66, align 4, !tbaa !6
  %68 = icmp eq i32 %67, 8
  br i1 %68, label %73, label %69

69:                                               ; preds = %65
  %70 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 8, i32 noundef %67, i32 noundef 8)
  %71 = load i32, ptr @errors, align 4, !tbaa !6
  %72 = add nsw i32 %71, 1
  store i32 %72, ptr @errors, align 4, !tbaa !6
  br label %73

73:                                               ; preds = %69, %65
  %74 = getelementptr inbounds nuw i8, ptr %1, i64 36
  %75 = load i32, ptr %74, align 4, !tbaa !6
  %76 = icmp eq i32 %75, 9
  br i1 %76, label %81, label %77

77:                                               ; preds = %73
  %78 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 9, i32 noundef %75, i32 noundef 9)
  %79 = load i32, ptr @errors, align 4, !tbaa !6
  %80 = add nsw i32 %79, 1
  store i32 %80, ptr @errors, align 4, !tbaa !6
  br label %81

81:                                               ; preds = %77, %73
  %82 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %83 = load i32, ptr %82, align 4, !tbaa !6
  %84 = icmp eq i32 %83, 10
  br i1 %84, label %89, label %85

85:                                               ; preds = %81
  %86 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, ptr noundef %0, i32 noundef 10, i32 noundef %83, i32 noundef 10)
  %87 = load i32, ptr @errors, align 4, !tbaa !6
  %88 = add nsw i32 %87, 1
  store i32 %88, ptr @errors, align 4, !tbaa !6
  br label %89

89:                                               ; preds = %85, %81
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #6

attributes #0 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold noreturn nounwind }
attributes #8 = { noreturn nounwind }
attributes #9 = { nounwind }

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
