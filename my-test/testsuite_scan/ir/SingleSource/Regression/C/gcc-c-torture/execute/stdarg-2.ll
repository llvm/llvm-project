; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stdarg-2.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stdarg-2.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }
%struct.A = type { i32, %struct.__va_list, [2 x %struct.__va_list] }

@foo_arg = dso_local local_unnamed_addr global i32 0, align 4
@gap = dso_local global %struct.__va_list zeroinitializer, align 8
@bar_arg = dso_local local_unnamed_addr global i32 0, align 4
@x = dso_local local_unnamed_addr global i64 0, align 8
@d = dso_local local_unnamed_addr global double 0.000000e+00, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(i32 noundef %0, ptr dead_on_return noundef captures(none) %1) local_unnamed_addr #0 {
  switch i32 %0, label %139 [
    i32 5, label %3
    i32 8, label %60
    i32 11, label %99
  ]

3:                                                ; preds = %2
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %5 = load i32, ptr %4, align 8
  %6 = icmp sgt i32 %5, -1
  br i1 %6, label %15, label %7

7:                                                ; preds = %3
  %8 = add nsw i32 %5, 8
  store i32 %8, ptr %4, align 8
  %9 = icmp samesign ult i32 %5, -7
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load ptr, ptr %11, align 8
  %13 = sext i32 %5 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %19

15:                                               ; preds = %7, %3
  %16 = phi i32 [ %8, %7 ], [ %5, %3 ]
  %17 = load ptr, ptr %1, align 8
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 8
  store ptr %18, ptr %1, align 8
  br label %19

19:                                               ; preds = %15, %10
  %20 = phi i32 [ %8, %10 ], [ %16, %15 ]
  %21 = phi ptr [ %14, %10 ], [ %17, %15 ]
  %22 = load i32, ptr %21, align 8, !tbaa !6
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %24 = load i32, ptr %23, align 4
  %25 = icmp sgt i32 %24, -1
  br i1 %25, label %34, label %26

26:                                               ; preds = %19
  %27 = add nsw i32 %24, 16
  store i32 %27, ptr %23, align 4
  %28 = icmp samesign ult i32 %24, -15
  br i1 %28, label %29, label %34

29:                                               ; preds = %26
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %31 = load ptr, ptr %30, align 8
  %32 = sext i32 %24 to i64
  %33 = getelementptr inbounds i8, ptr %31, i64 %32
  br label %37

34:                                               ; preds = %26, %19
  %35 = load ptr, ptr %1, align 8
  %36 = getelementptr inbounds nuw i8, ptr %35, i64 8
  store ptr %36, ptr %1, align 8
  br label %37

37:                                               ; preds = %34, %29
  %38 = phi ptr [ %33, %29 ], [ %35, %34 ]
  %39 = load double, ptr %38, align 8, !tbaa !10
  %40 = sitofp i32 %22 to double
  %41 = fadd double %39, %40
  %42 = fptosi double %41 to i32
  %43 = icmp sgt i32 %20, -1
  br i1 %43, label %52, label %44

44:                                               ; preds = %37
  %45 = add nsw i32 %20, 8
  store i32 %45, ptr %4, align 8
  %46 = icmp samesign ult i32 %20, -7
  br i1 %46, label %47, label %52

47:                                               ; preds = %44
  %48 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %49 = load ptr, ptr %48, align 8
  %50 = sext i32 %20 to i64
  %51 = getelementptr inbounds i8, ptr %49, i64 %50
  br label %55

52:                                               ; preds = %44, %37
  %53 = load ptr, ptr %1, align 8
  %54 = getelementptr inbounds nuw i8, ptr %53, i64 8
  store ptr %54, ptr %1, align 8
  br label %55

55:                                               ; preds = %52, %47
  %56 = phi ptr [ %51, %47 ], [ %53, %52 ]
  %57 = load i64, ptr %56, align 8, !tbaa !12
  %58 = trunc i64 %57 to i32
  %59 = add i32 %58, %42
  br label %140

60:                                               ; preds = %2
  %61 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %62 = load i32, ptr %61, align 8
  %63 = icmp sgt i32 %62, -1
  br i1 %63, label %72, label %64

64:                                               ; preds = %60
  %65 = add nsw i32 %62, 8
  store i32 %65, ptr %61, align 8
  %66 = icmp samesign ult i32 %62, -7
  br i1 %66, label %67, label %72

67:                                               ; preds = %64
  %68 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %69 = load ptr, ptr %68, align 8
  %70 = sext i32 %62 to i64
  %71 = getelementptr inbounds i8, ptr %69, i64 %70
  br label %75

72:                                               ; preds = %64, %60
  %73 = load ptr, ptr %1, align 8
  %74 = getelementptr inbounds nuw i8, ptr %73, i64 8
  store ptr %74, ptr %1, align 8
  br label %75

75:                                               ; preds = %72, %67
  %76 = phi ptr [ %71, %67 ], [ %73, %72 ]
  %77 = load i64, ptr %76, align 8, !tbaa !12
  %78 = trunc i64 %77 to i32
  %79 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %80 = load i32, ptr %79, align 4
  %81 = icmp sgt i32 %80, -1
  br i1 %81, label %90, label %82

82:                                               ; preds = %75
  %83 = add nsw i32 %80, 16
  store i32 %83, ptr %79, align 4
  %84 = icmp samesign ult i32 %80, -15
  br i1 %84, label %85, label %90

85:                                               ; preds = %82
  %86 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %87 = load ptr, ptr %86, align 8
  %88 = sext i32 %80 to i64
  %89 = getelementptr inbounds i8, ptr %87, i64 %88
  br label %93

90:                                               ; preds = %82, %75
  %91 = load ptr, ptr %1, align 8
  %92 = getelementptr inbounds nuw i8, ptr %91, i64 8
  store ptr %92, ptr %1, align 8
  br label %93

93:                                               ; preds = %90, %85
  %94 = phi ptr [ %89, %85 ], [ %91, %90 ]
  %95 = load double, ptr %94, align 8, !tbaa !10
  %96 = sitofp i32 %78 to double
  %97 = fadd double %95, %96
  %98 = fptosi double %97 to i32
  br label %140

99:                                               ; preds = %2
  %100 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %101 = load i32, ptr %100, align 8
  %102 = icmp sgt i32 %101, -1
  br i1 %102, label %111, label %103

103:                                              ; preds = %99
  %104 = add nsw i32 %101, 8
  store i32 %104, ptr %100, align 8
  %105 = icmp samesign ult i32 %101, -7
  br i1 %105, label %106, label %111

106:                                              ; preds = %103
  %107 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %108 = load ptr, ptr %107, align 8
  %109 = sext i32 %101 to i64
  %110 = getelementptr inbounds i8, ptr %108, i64 %109
  br label %114

111:                                              ; preds = %103, %99
  %112 = load ptr, ptr %1, align 8
  %113 = getelementptr inbounds nuw i8, ptr %112, i64 8
  store ptr %113, ptr %1, align 8
  br label %114

114:                                              ; preds = %111, %106
  %115 = phi ptr [ %110, %106 ], [ %112, %111 ]
  %116 = load i32, ptr %115, align 8, !tbaa !6
  %117 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %118 = load i32, ptr %117, align 4
  %119 = icmp sgt i32 %118, -1
  br i1 %119, label %128, label %120

120:                                              ; preds = %114
  %121 = add nsw i32 %118, 16
  store i32 %121, ptr %117, align 4
  %122 = icmp samesign ult i32 %118, -15
  br i1 %122, label %123, label %128

123:                                              ; preds = %120
  %124 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %125 = load ptr, ptr %124, align 8
  %126 = sext i32 %118 to i64
  %127 = getelementptr inbounds i8, ptr %125, i64 %126
  br label %133

128:                                              ; preds = %120, %114
  %129 = load ptr, ptr %1, align 8
  %130 = getelementptr inbounds nuw i8, ptr %129, i64 15
  %131 = tail call align 16 ptr @llvm.ptrmask.p0.i64(ptr nonnull %130, i64 -16)
  %132 = getelementptr inbounds nuw i8, ptr %131, i64 16
  store ptr %132, ptr %1, align 8
  br label %133

133:                                              ; preds = %128, %123
  %134 = phi ptr [ %127, %123 ], [ %131, %128 ]
  %135 = load fp128, ptr %134, align 16, !tbaa !14
  %136 = sitofp i32 %116 to fp128
  %137 = fadd fp128 %135, %136
  %138 = fptosi fp128 %137 to i32
  br label %140

139:                                              ; preds = %2
  tail call void @abort() #7
  unreachable

140:                                              ; preds = %133, %93, %55
  %141 = phi i32 [ %138, %133 ], [ %98, %93 ], [ %59, %55 ]
  store i32 %141, ptr @foo_arg, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.ptrmask.p0.i64(ptr, i64) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(i32 noundef %0) local_unnamed_addr #0 {
  %2 = icmp eq i32 %0, 16386
  br i1 %2, label %3, label %38

3:                                                ; preds = %1
  %4 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %13, label %6

6:                                                ; preds = %3
  %7 = add nsw i32 %4, 8
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %8 = icmp samesign ult i32 %4, -7
  br i1 %8, label %9, label %13

9:                                                ; preds = %6
  %10 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 8), align 8
  %11 = sext i32 %4 to i64
  %12 = getelementptr inbounds i8, ptr %10, i64 %11
  br label %16

13:                                               ; preds = %6, %3
  %14 = load ptr, ptr @gap, align 8
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 8
  store ptr %15, ptr @gap, align 8
  br label %16

16:                                               ; preds = %13, %9
  %17 = phi ptr [ %12, %9 ], [ %14, %13 ]
  %18 = load i32, ptr %17, align 8, !tbaa !6
  %19 = icmp eq i32 %18, 13
  br i1 %19, label %20, label %37

20:                                               ; preds = %16
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %22 = icmp sgt i32 %21, -1
  br i1 %22, label %30, label %23

23:                                               ; preds = %20
  %24 = add nsw i32 %21, 16
  store i32 %24, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %25 = icmp samesign ult i32 %21, -15
  br i1 %25, label %26, label %30

26:                                               ; preds = %23
  %27 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 16), align 8
  %28 = sext i32 %21 to i64
  %29 = getelementptr inbounds i8, ptr %27, i64 %28
  br label %33

30:                                               ; preds = %23, %20
  %31 = load ptr, ptr @gap, align 8
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 8
  store ptr %32, ptr @gap, align 8
  br label %33

33:                                               ; preds = %30, %26
  %34 = phi ptr [ %29, %26 ], [ %31, %30 ]
  %35 = load double, ptr %34, align 8, !tbaa !10
  %36 = fcmp une double %35, -1.400000e+01
  br i1 %36, label %37, label %38

37:                                               ; preds = %33, %16
  tail call void @abort() #7
  unreachable

38:                                               ; preds = %33, %1
  store i32 %0, ptr @bar_arg, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define dso_local void @f1(i32 %0, ...) local_unnamed_addr #3 {
  tail call void @llvm.va_start.p0(ptr nonnull @gap)
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %3 = icmp sgt i32 %2, -1
  br i1 %3, label %11, label %4

4:                                                ; preds = %1
  %5 = add nsw i32 %2, 8
  store i32 %5, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %6 = icmp samesign ult i32 %2, -7
  br i1 %6, label %7, label %11

7:                                                ; preds = %4
  %8 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 8), align 8
  %9 = sext i32 %2 to i64
  %10 = getelementptr inbounds i8, ptr %8, i64 %9
  br label %14

11:                                               ; preds = %4, %1
  %12 = load ptr, ptr @gap, align 8
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store ptr %13, ptr @gap, align 8
  br label %14

14:                                               ; preds = %11, %7
  %15 = phi ptr [ %10, %7 ], [ %12, %11 ]
  %16 = load i64, ptr %15, align 8, !tbaa !16
  store i64 %16, ptr @x, align 8, !tbaa !16
  tail call void @llvm.va_end.p0(ptr @gap)
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #4

; Function Attrs: nofree nounwind uwtable
define dso_local void @f2(i32 noundef %0, ...) local_unnamed_addr #0 {
  tail call void @llvm.va_start.p0(ptr nonnull @gap)
  %2 = icmp eq i32 %0, 16386
  br i1 %2, label %3, label %38

3:                                                ; preds = %1
  %4 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %13, label %6

6:                                                ; preds = %3
  %7 = add nsw i32 %4, 8
  store i32 %7, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %8 = icmp samesign ult i32 %4, -7
  br i1 %8, label %9, label %13

9:                                                ; preds = %6
  %10 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 8), align 8
  %11 = sext i32 %4 to i64
  %12 = getelementptr inbounds i8, ptr %10, i64 %11
  br label %16

13:                                               ; preds = %6, %3
  %14 = load ptr, ptr @gap, align 8
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 8
  store ptr %15, ptr @gap, align 8
  br label %16

16:                                               ; preds = %13, %9
  %17 = phi ptr [ %12, %9 ], [ %14, %13 ]
  %18 = load i32, ptr %17, align 8, !tbaa !6
  %19 = icmp eq i32 %18, 13
  br i1 %19, label %20, label %37

20:                                               ; preds = %16
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %22 = icmp sgt i32 %21, -1
  br i1 %22, label %30, label %23

23:                                               ; preds = %20
  %24 = add nsw i32 %21, 16
  store i32 %24, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %25 = icmp samesign ult i32 %21, -15
  br i1 %25, label %26, label %30

26:                                               ; preds = %23
  %27 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 16), align 8
  %28 = sext i32 %21 to i64
  %29 = getelementptr inbounds i8, ptr %27, i64 %28
  br label %33

30:                                               ; preds = %23, %20
  %31 = load ptr, ptr @gap, align 8
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 8
  store ptr %32, ptr @gap, align 8
  br label %33

33:                                               ; preds = %30, %26
  %34 = phi ptr [ %29, %26 ], [ %31, %30 ]
  %35 = load double, ptr %34, align 8, !tbaa !10
  %36 = fcmp une double %35, -1.400000e+01
  br i1 %36, label %37, label %38

37:                                               ; preds = %33, %16
  tail call void @abort() #7
  unreachable

38:                                               ; preds = %1, %33
  store i32 %0, ptr @bar_arg, align 4, !tbaa !6
  tail call void @llvm.va_end.p0(ptr @gap)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define dso_local void @f3(i32 %0, ...) local_unnamed_addr #3 {
  %2 = alloca [10 x %struct.__va_list], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 128
  call void @llvm.va_start.p0(ptr nonnull %3)
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 152
  %5 = load i32, ptr %4, align 8
  %6 = icmp sgt i32 %5, -1
  br i1 %6, label %15, label %7

7:                                                ; preds = %1
  %8 = add nsw i32 %5, 8
  store i32 %8, ptr %4, align 8
  %9 = icmp samesign ult i32 %5, -7
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 136
  %12 = load ptr, ptr %11, align 8
  %13 = sext i32 %5 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %18

15:                                               ; preds = %7, %1
  %16 = load ptr, ptr %3, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %17, ptr %3, align 8
  br label %18

18:                                               ; preds = %15, %10
  %19 = phi ptr [ %14, %10 ], [ %16, %15 ]
  %20 = load i64, ptr %19, align 8, !tbaa !16
  store i64 %20, ptr @x, align 8, !tbaa !16
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #5

; Function Attrs: nofree nounwind uwtable
define dso_local void @f4(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca [10 x %struct.__va_list], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 128
  call void @llvm.va_start.p0(ptr nonnull %3)
  %4 = icmp eq i32 %0, 16386
  br i1 %4, label %5, label %40

5:                                                ; preds = %1
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %7 = icmp sgt i32 %6, -1
  br i1 %7, label %15, label %8

8:                                                ; preds = %5
  %9 = add nsw i32 %6, 8
  store i32 %9, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %10 = icmp samesign ult i32 %6, -7
  br i1 %10, label %11, label %15

11:                                               ; preds = %8
  %12 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 8), align 8
  %13 = sext i32 %6 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %18

15:                                               ; preds = %8, %5
  %16 = load ptr, ptr @gap, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %17, ptr @gap, align 8
  br label %18

18:                                               ; preds = %15, %11
  %19 = phi ptr [ %14, %11 ], [ %16, %15 ]
  %20 = load i32, ptr %19, align 8, !tbaa !6
  %21 = icmp eq i32 %20, 13
  br i1 %21, label %22, label %39

22:                                               ; preds = %18
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %24 = icmp sgt i32 %23, -1
  br i1 %24, label %32, label %25

25:                                               ; preds = %22
  %26 = add nsw i32 %23, 16
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %27 = icmp samesign ult i32 %23, -15
  br i1 %27, label %28, label %32

28:                                               ; preds = %25
  %29 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 16), align 8
  %30 = sext i32 %23 to i64
  %31 = getelementptr inbounds i8, ptr %29, i64 %30
  br label %35

32:                                               ; preds = %25, %22
  %33 = load ptr, ptr @gap, align 8
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 8
  store ptr %34, ptr @gap, align 8
  br label %35

35:                                               ; preds = %32, %28
  %36 = phi ptr [ %31, %28 ], [ %33, %32 ]
  %37 = load double, ptr %36, align 8, !tbaa !10
  %38 = fcmp une double %37, -1.400000e+01
  br i1 %38, label %39, label %40

39:                                               ; preds = %35, %18
  call void @abort() #7
  unreachable

40:                                               ; preds = %1, %35
  store i32 %0, ptr @bar_arg, align 4, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @f5(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca [10 x %struct.__va_list], align 8
  %3 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 128
  call void @llvm.va_start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %4, i64 32, i1 false), !tbaa.struct !18
  call void @foo(i32 noundef %0, ptr dead_on_return noundef nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  call void @llvm.va_end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #6

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define dso_local void @f6(i32 %0, ...) local_unnamed_addr #3 {
  %2 = alloca %struct.A, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  call void @llvm.va_start.p0(ptr nonnull %3)
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 32
  %5 = load i32, ptr %4, align 8
  %6 = icmp sgt i32 %5, -1
  br i1 %6, label %15, label %7

7:                                                ; preds = %1
  %8 = add nsw i32 %5, 8
  store i32 %8, ptr %4, align 8
  %9 = icmp samesign ult i32 %5, -7
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %12 = load ptr, ptr %11, align 8
  %13 = sext i32 %5 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %18

15:                                               ; preds = %7, %1
  %16 = load ptr, ptr %3, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %17, ptr %3, align 8
  br label %18

18:                                               ; preds = %15, %10
  %19 = phi ptr [ %14, %10 ], [ %16, %15 ]
  %20 = load i64, ptr %19, align 8, !tbaa !16
  store i64 %20, ptr @x, align 8, !tbaa !16
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @f7(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.A, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  call void @llvm.va_start.p0(ptr nonnull %3)
  %4 = icmp eq i32 %0, 16386
  br i1 %4, label %5, label %40

5:                                                ; preds = %1
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %7 = icmp sgt i32 %6, -1
  br i1 %7, label %15, label %8

8:                                                ; preds = %5
  %9 = add nsw i32 %6, 8
  store i32 %9, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %10 = icmp samesign ult i32 %6, -7
  br i1 %10, label %11, label %15

11:                                               ; preds = %8
  %12 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 8), align 8
  %13 = sext i32 %6 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %18

15:                                               ; preds = %8, %5
  %16 = load ptr, ptr @gap, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %17, ptr @gap, align 8
  br label %18

18:                                               ; preds = %15, %11
  %19 = phi ptr [ %14, %11 ], [ %16, %15 ]
  %20 = load i32, ptr %19, align 8, !tbaa !6
  %21 = icmp eq i32 %20, 13
  br i1 %21, label %22, label %39

22:                                               ; preds = %18
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %24 = icmp sgt i32 %23, -1
  br i1 %24, label %32, label %25

25:                                               ; preds = %22
  %26 = add nsw i32 %23, 16
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %27 = icmp samesign ult i32 %23, -15
  br i1 %27, label %28, label %32

28:                                               ; preds = %25
  %29 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 16), align 8
  %30 = sext i32 %23 to i64
  %31 = getelementptr inbounds i8, ptr %29, i64 %30
  br label %35

32:                                               ; preds = %25, %22
  %33 = load ptr, ptr @gap, align 8
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 8
  store ptr %34, ptr @gap, align 8
  br label %35

35:                                               ; preds = %32, %28
  %36 = phi ptr [ %31, %28 ], [ %33, %32 ]
  %37 = load double, ptr %36, align 8, !tbaa !10
  %38 = fcmp une double %37, -1.400000e+01
  br i1 %38, label %39, label %40

39:                                               ; preds = %35, %18
  call void @abort() #7
  unreachable

40:                                               ; preds = %1, %35
  store i32 %0, ptr @bar_arg, align 4, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @f8(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.A, align 8
  %3 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 8
  call void @llvm.va_start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %4, i64 32, i1 false), !tbaa.struct !18
  call void @foo(i32 noundef %0, ptr dead_on_return noundef nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  call void @llvm.va_end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define dso_local void @f10(i32 %0, ...) local_unnamed_addr #3 {
  %2 = alloca %struct.A, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 72
  call void @llvm.va_start.p0(ptr nonnull %3)
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 96
  %5 = load i32, ptr %4, align 8
  %6 = icmp sgt i32 %5, -1
  br i1 %6, label %15, label %7

7:                                                ; preds = %1
  %8 = add nsw i32 %5, 8
  store i32 %8, ptr %4, align 8
  %9 = icmp samesign ult i32 %5, -7
  br i1 %9, label %10, label %15

10:                                               ; preds = %7
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 80
  %12 = load ptr, ptr %11, align 8
  %13 = sext i32 %5 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %18

15:                                               ; preds = %7, %1
  %16 = load ptr, ptr %3, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %17, ptr %3, align 8
  br label %18

18:                                               ; preds = %15, %10
  %19 = phi ptr [ %14, %10 ], [ %16, %15 ]
  %20 = load i64, ptr %19, align 8, !tbaa !16
  store i64 %20, ptr @x, align 8, !tbaa !16
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @f11(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.A, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 72
  call void @llvm.va_start.p0(ptr nonnull %3)
  %4 = icmp eq i32 %0, 16386
  br i1 %4, label %5, label %40

5:                                                ; preds = %1
  %6 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %7 = icmp sgt i32 %6, -1
  br i1 %7, label %15, label %8

8:                                                ; preds = %5
  %9 = add nsw i32 %6, 8
  store i32 %9, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %10 = icmp samesign ult i32 %6, -7
  br i1 %10, label %11, label %15

11:                                               ; preds = %8
  %12 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 8), align 8
  %13 = sext i32 %6 to i64
  %14 = getelementptr inbounds i8, ptr %12, i64 %13
  br label %18

15:                                               ; preds = %8, %5
  %16 = load ptr, ptr @gap, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store ptr %17, ptr @gap, align 8
  br label %18

18:                                               ; preds = %15, %11
  %19 = phi ptr [ %14, %11 ], [ %16, %15 ]
  %20 = load i32, ptr %19, align 8, !tbaa !6
  %21 = icmp eq i32 %20, 13
  br i1 %21, label %22, label %39

22:                                               ; preds = %18
  %23 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %24 = icmp sgt i32 %23, -1
  br i1 %24, label %32, label %25

25:                                               ; preds = %22
  %26 = add nsw i32 %23, 16
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %27 = icmp samesign ult i32 %23, -15
  br i1 %27, label %28, label %32

28:                                               ; preds = %25
  %29 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 16), align 8
  %30 = sext i32 %23 to i64
  %31 = getelementptr inbounds i8, ptr %29, i64 %30
  br label %35

32:                                               ; preds = %25, %22
  %33 = load ptr, ptr @gap, align 8
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 8
  store ptr %34, ptr @gap, align 8
  br label %35

35:                                               ; preds = %32, %28
  %36 = phi ptr [ %31, %28 ], [ %33, %32 ]
  %37 = load double, ptr %36, align 8, !tbaa !10
  %38 = fcmp une double %37, -1.400000e+01
  br i1 %38, label %39, label %40

39:                                               ; preds = %35, %18
  call void @abort() #7
  unreachable

40:                                               ; preds = %1, %35
  store i32 %0, ptr @bar_arg, align 4, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @f12(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.A, align 8
  %3 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 72
  call void @llvm.va_start.p0(ptr nonnull %4)
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) %3, ptr noundef nonnull align 8 dereferenceable(32) %4, i64 32, i1 false), !tbaa.struct !18
  call void @foo(i32 noundef %0, ptr dead_on_return noundef nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #8
  call void @llvm.va_end.p0(ptr nonnull %4)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  tail call void (i32, ...) @f1(i32 poison, i64 noundef 79)
  %1 = load i64, ptr @x, align 8, !tbaa !16
  %2 = icmp eq i64 %1, 79
  br i1 %2, label %4, label %3

3:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

4:                                                ; preds = %0
  tail call void (i32, ...) @f2(i32 noundef 16386, i32 noundef 13, double noundef -1.400000e+01)
  %5 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %6 = icmp eq i32 %5, 16386
  br i1 %6, label %8, label %7

7:                                                ; preds = %4
  tail call void @abort() #7
  unreachable

8:                                                ; preds = %4
  tail call void (i32, ...) @f3(i32 poison, i64 noundef 2031)
  %9 = load i64, ptr @x, align 8, !tbaa !16
  %10 = icmp eq i64 %9, 2031
  br i1 %10, label %12, label %11

11:                                               ; preds = %8
  tail call void @abort() #7
  unreachable

12:                                               ; preds = %8
  tail call void (i32, ...) @f4(i32 noundef 4, i32 noundef 18)
  %13 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %14 = icmp eq i32 %13, 4
  br i1 %14, label %16, label %15

15:                                               ; preds = %12
  tail call void @abort() #7
  unreachable

16:                                               ; preds = %12
  tail call void (i32, ...) @f5(i32 noundef 5, i32 noundef 1, double noundef 1.900000e+01, i64 noundef 18)
  %17 = load i32, ptr @foo_arg, align 4, !tbaa !6
  %18 = icmp eq i32 %17, 38
  br i1 %18, label %20, label %19

19:                                               ; preds = %16
  tail call void @abort() #7
  unreachable

20:                                               ; preds = %16
  tail call void (i32, ...) @f6(i32 poison, i64 noundef 18)
  %21 = load i64, ptr @x, align 8, !tbaa !16
  %22 = icmp eq i64 %21, 18
  br i1 %22, label %24, label %23

23:                                               ; preds = %20
  tail call void @abort() #7
  unreachable

24:                                               ; preds = %20
  tail call void (i32, ...) @f7(i32 noundef 7)
  %25 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %26 = icmp eq i32 %25, 7
  br i1 %26, label %28, label %27

27:                                               ; preds = %24
  tail call void @abort() #7
  unreachable

28:                                               ; preds = %24
  tail call void (i32, ...) @f8(i32 noundef 8, i64 noundef 2031, double noundef 1.300000e+01)
  %29 = load i32, ptr @foo_arg, align 4, !tbaa !6
  %30 = icmp eq i32 %29, 2044
  br i1 %30, label %32, label %31

31:                                               ; preds = %28
  tail call void @abort() #7
  unreachable

32:                                               ; preds = %28
  tail call void (i32, ...) @f10(i32 poison, i64 noundef 180)
  %33 = load i64, ptr @x, align 8, !tbaa !16
  %34 = icmp eq i64 %33, 180
  br i1 %34, label %36, label %35

35:                                               ; preds = %32
  tail call void @abort() #7
  unreachable

36:                                               ; preds = %32
  tail call void (i32, ...) @f11(i32 noundef 10)
  %37 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %38 = icmp eq i32 %37, 10
  br i1 %38, label %40, label %39

39:                                               ; preds = %36
  tail call void @abort() #7
  unreachable

40:                                               ; preds = %36
  tail call void (i32, ...) @f12(i32 noundef 11, i32 noundef 2030, fp128 noundef 0xL00000000000000004002800000000000)
  %41 = load i32, ptr @foo_arg, align 4, !tbaa !6
  %42 = icmp eq i32 %41, 2042
  br i1 %42, label %44, label %43

43:                                               ; preds = %40
  tail call void @abort() #7
  unreachable

44:                                               ; preds = %40
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { noreturn nounwind }
attributes #8 = { nounwind }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"double", !8, i64 0}
!12 = !{!13, !13, i64 0}
!13 = !{!"long long", !8, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"long double", !8, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"long", !8, i64 0}
!18 = !{i64 0, i64 8, !19, i64 8, i64 8, !19, i64 16, i64 8, !19, i64 24, i64 4, !6, i64 28, i64 4, !6}
!19 = !{!20, !20, i64 0}
!20 = !{!"any pointer", !8, i64 0}
