; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stdarg-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stdarg-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@foo_arg = dso_local local_unnamed_addr global i32 0, align 4
@gap = dso_local global %struct.__va_list zeroinitializer, align 8
@pap = dso_local local_unnamed_addr global ptr null, align 8
@bar_arg = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@x = dso_local local_unnamed_addr global i64 0, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo(i32 noundef %0, ptr dead_on_return noundef captures(none) %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 5
  br i1 %3, label %4, label %22

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %6 = load i32, ptr %5, align 8
  %7 = icmp sgt i32 %6, -1
  br i1 %7, label %16, label %8

8:                                                ; preds = %4
  %9 = add nsw i32 %6, 8
  store i32 %9, ptr %5, align 8
  %10 = icmp samesign ult i32 %6, -7
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %13 = load ptr, ptr %12, align 8
  %14 = sext i32 %6 to i64
  %15 = getelementptr inbounds i8, ptr %13, i64 %14
  br label %19

16:                                               ; preds = %8, %4
  %17 = load ptr, ptr %1, align 8
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 8
  store ptr %18, ptr %1, align 8
  br label %19

19:                                               ; preds = %16, %11
  %20 = phi ptr [ %15, %11 ], [ %17, %16 ]
  %21 = load i32, ptr %20, align 8, !tbaa !6
  store i32 %21, ptr @foo_arg, align 4, !tbaa !6
  ret void

22:                                               ; preds = %2
  tail call void @abort() #7
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(i32 noundef %0) local_unnamed_addr #0 {
  switch i32 %0, label %100 [
    i32 16390, label %2
    i32 16392, label %37
  ]

2:                                                ; preds = %1
  %3 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %4 = icmp sgt i32 %3, -1
  br i1 %4, label %12, label %5

5:                                                ; preds = %2
  %6 = add nsw i32 %3, 16
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 28), align 4
  %7 = icmp samesign ult i32 %3, -15
  br i1 %7, label %8, label %12

8:                                                ; preds = %5
  %9 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 16), align 8
  %10 = sext i32 %3 to i64
  %11 = getelementptr inbounds i8, ptr %9, i64 %10
  br label %15

12:                                               ; preds = %5, %2
  %13 = load ptr, ptr @gap, align 8
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store ptr %14, ptr @gap, align 8
  br label %15

15:                                               ; preds = %12, %8
  %16 = phi ptr [ %11, %8 ], [ %13, %12 ]
  %17 = load double, ptr %16, align 8, !tbaa !10
  %18 = fcmp une double %17, 1.700000e+01
  br i1 %18, label %36, label %19

19:                                               ; preds = %15
  %20 = load i32, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %21 = icmp sgt i32 %20, -1
  br i1 %21, label %29, label %22

22:                                               ; preds = %19
  %23 = add nsw i32 %20, 8
  store i32 %23, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 24), align 8
  %24 = icmp samesign ult i32 %20, -7
  br i1 %24, label %25, label %29

25:                                               ; preds = %22
  %26 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @gap, i64 8), align 8
  %27 = sext i32 %20 to i64
  %28 = getelementptr inbounds i8, ptr %26, i64 %27
  br label %32

29:                                               ; preds = %22, %19
  %30 = load ptr, ptr @gap, align 8
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 8
  store ptr %31, ptr @gap, align 8
  br label %32

32:                                               ; preds = %29, %25
  %33 = phi ptr [ %28, %25 ], [ %30, %29 ]
  %34 = load i64, ptr %33, align 8, !tbaa !12
  %35 = icmp eq i64 %34, 129
  br i1 %35, label %100, label %36

36:                                               ; preds = %32, %15
  tail call void @abort() #7
  unreachable

37:                                               ; preds = %1
  %38 = load ptr, ptr @pap, align 8, !tbaa !14
  %39 = getelementptr inbounds nuw i8, ptr %38, i64 24
  %40 = load i32, ptr %39, align 8
  %41 = icmp sgt i32 %40, -1
  br i1 %41, label %50, label %42

42:                                               ; preds = %37
  %43 = add nsw i32 %40, 8
  store i32 %43, ptr %39, align 8
  %44 = icmp samesign ult i32 %40, -7
  br i1 %44, label %45, label %50

45:                                               ; preds = %42
  %46 = getelementptr inbounds nuw i8, ptr %38, i64 8
  %47 = load ptr, ptr %46, align 8
  %48 = sext i32 %40 to i64
  %49 = getelementptr inbounds i8, ptr %47, i64 %48
  br label %53

50:                                               ; preds = %42, %37
  %51 = load ptr, ptr %38, align 8
  %52 = getelementptr inbounds nuw i8, ptr %51, i64 8
  store ptr %52, ptr %38, align 8
  br label %53

53:                                               ; preds = %50, %45
  %54 = phi ptr [ %49, %45 ], [ %51, %50 ]
  %55 = load i64, ptr %54, align 8, !tbaa !17
  %56 = icmp eq i64 %55, 14
  br i1 %56, label %57, label %99

57:                                               ; preds = %53
  %58 = load ptr, ptr @pap, align 8, !tbaa !14
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 28
  %60 = load i32, ptr %59, align 4
  %61 = icmp sgt i32 %60, -1
  br i1 %61, label %70, label %62

62:                                               ; preds = %57
  %63 = add nsw i32 %60, 16
  store i32 %63, ptr %59, align 4
  %64 = icmp samesign ult i32 %60, -15
  br i1 %64, label %65, label %70

65:                                               ; preds = %62
  %66 = getelementptr inbounds nuw i8, ptr %58, i64 16
  %67 = load ptr, ptr %66, align 8
  %68 = sext i32 %60 to i64
  %69 = getelementptr inbounds i8, ptr %67, i64 %68
  br label %75

70:                                               ; preds = %62, %57
  %71 = load ptr, ptr %58, align 8
  %72 = getelementptr inbounds nuw i8, ptr %71, i64 15
  %73 = tail call align 16 ptr @llvm.ptrmask.p0.i64(ptr nonnull %72, i64 -16)
  %74 = getelementptr inbounds nuw i8, ptr %73, i64 16
  store ptr %74, ptr %58, align 8
  br label %75

75:                                               ; preds = %70, %65
  %76 = phi ptr [ %69, %65 ], [ %73, %70 ]
  %77 = load fp128, ptr %76, align 16, !tbaa !19
  %78 = fcmp une fp128 %77, 0xL00000000000000004006060000000000
  br i1 %78, label %99, label %79

79:                                               ; preds = %75
  %80 = load ptr, ptr @pap, align 8, !tbaa !14
  %81 = getelementptr inbounds nuw i8, ptr %80, i64 24
  %82 = load i32, ptr %81, align 8
  %83 = icmp sgt i32 %82, -1
  br i1 %83, label %92, label %84

84:                                               ; preds = %79
  %85 = add nsw i32 %82, 8
  store i32 %85, ptr %81, align 8
  %86 = icmp samesign ult i32 %82, -7
  br i1 %86, label %87, label %92

87:                                               ; preds = %84
  %88 = getelementptr inbounds nuw i8, ptr %80, i64 8
  %89 = load ptr, ptr %88, align 8
  %90 = sext i32 %82 to i64
  %91 = getelementptr inbounds i8, ptr %89, i64 %90
  br label %95

92:                                               ; preds = %84, %79
  %93 = load ptr, ptr %80, align 8
  %94 = getelementptr inbounds nuw i8, ptr %93, i64 8
  store ptr %94, ptr %80, align 8
  br label %95

95:                                               ; preds = %92, %87
  %96 = phi ptr [ %91, %87 ], [ %93, %92 ]
  %97 = load i32, ptr %96, align 8, !tbaa !6
  %98 = icmp eq i32 %97, 17
  br i1 %98, label %100, label %99

99:                                               ; preds = %95, %75, %53
  tail call void @abort() #7
  unreachable

100:                                              ; preds = %1, %95, %32
  store i32 %0, ptr @bar_arg, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare ptr @llvm.ptrmask.p0.i64(ptr, i64) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @f0(i32 noundef %0, ...) local_unnamed_addr #3 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @f1(i32 noundef %0, ...) local_unnamed_addr #3 {
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

; Function Attrs: nofree nounwind uwtable
define dso_local void @f2(i32 %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = load double, ptr @d, align 8, !tbaa !10
  %4 = fptosi double %3 to i32
  call void @bar(i32 noundef %4)
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %6 = load i32, ptr %5, align 8
  %7 = icmp sgt i32 %6, -1
  br i1 %7, label %16, label %8

8:                                                ; preds = %1
  %9 = add nsw i32 %6, 8
  store i32 %9, ptr %5, align 8
  %10 = icmp samesign ult i32 %6, -7
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %13 = load ptr, ptr %12, align 8
  %14 = sext i32 %6 to i64
  %15 = getelementptr inbounds i8, ptr %13, i64 %14
  br label %19

16:                                               ; preds = %8, %1
  %17 = load ptr, ptr %2, align 8
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 8
  store ptr %18, ptr %2, align 8
  br label %19

19:                                               ; preds = %16, %11
  %20 = phi ptr [ %15, %11 ], [ %17, %16 ]
  %21 = load i64, ptr %20, align 8, !tbaa !12
  store i64 %21, ptr @x, align 8, !tbaa !12
  %22 = trunc i64 %21 to i32
  call void @bar(i32 noundef %22)
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define dso_local void @f3(i32 %0, ...) local_unnamed_addr #6 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %4 = load i32, ptr %3, align 4
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %14, label %6

6:                                                ; preds = %1
  %7 = add nsw i32 %4, 16
  store i32 %7, ptr %3, align 4
  %8 = icmp samesign ult i32 %4, -15
  br i1 %8, label %9, label %14

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %11 = load ptr, ptr %10, align 8
  %12 = sext i32 %4 to i64
  %13 = getelementptr inbounds i8, ptr %11, i64 %12
  br label %17

14:                                               ; preds = %6, %1
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store ptr %16, ptr %2, align 8
  br label %17

17:                                               ; preds = %14, %9
  %18 = phi ptr [ %13, %9 ], [ %15, %14 ]
  %19 = load double, ptr %18, align 8, !tbaa !10
  store double %19, ptr @d, align 8, !tbaa !10
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @f4(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %4 = load i32, ptr %3, align 4
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %14, label %6

6:                                                ; preds = %1
  %7 = add nsw i32 %4, 16
  store i32 %7, ptr %3, align 4
  %8 = icmp samesign ult i32 %4, -15
  br i1 %8, label %9, label %14

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %11 = load ptr, ptr %10, align 8
  %12 = sext i32 %4 to i64
  %13 = getelementptr inbounds i8, ptr %11, i64 %12
  br label %17

14:                                               ; preds = %6, %1
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store ptr %16, ptr %2, align 8
  br label %17

17:                                               ; preds = %14, %9
  %18 = phi ptr [ %13, %9 ], [ %15, %14 ]
  %19 = load double, ptr %18, align 8, !tbaa !10
  %20 = fptosi double %19 to i64
  store i64 %20, ptr @x, align 8, !tbaa !12
  %21 = icmp eq i32 %0, 5
  br i1 %21, label %22, label %33

22:                                               ; preds = %17
  %23 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %24 = load i32, ptr %23, align 8, !tbaa !6
  %25 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %26 = load ptr, ptr %25, align 8, !tbaa !21
  %27 = load ptr, ptr %2, align 8, !tbaa !21
  %28 = icmp slt i32 %24, -7
  %29 = sext i32 %24 to i64
  %30 = getelementptr inbounds i8, ptr %26, i64 %29
  %31 = select i1 %28, ptr %30, ptr %27
  %32 = load i32, ptr %31, align 8, !tbaa !6
  store i32 %32, ptr @foo_arg, align 4, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void

33:                                               ; preds = %17
  call void @abort() #7
  unreachable
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @f5(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.va_start.p0(ptr nonnull %2)
  call void @llvm.va_copy.p0(ptr nonnull @gap, ptr nonnull %2)
  call void @bar(i32 noundef %0)
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.va_end.p0(ptr @gap)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_copy.p0(ptr, ptr) #5

; Function Attrs: nofree nounwind uwtable
define dso_local void @f6(i32 %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = load double, ptr @d, align 8, !tbaa !10
  %4 = fptosi double %3 to i32
  call void @bar(i32 noundef %4)
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %6 = load i32, ptr %5, align 8
  %7 = icmp sgt i32 %6, -1
  br i1 %7, label %11, label %8

8:                                                ; preds = %1
  %9 = add nsw i32 %6, 8
  store i32 %9, ptr %5, align 8
  %10 = icmp samesign ult i32 %6, -7
  br i1 %10, label %14, label %11

11:                                               ; preds = %1, %8
  %12 = load ptr, ptr %2, align 8
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store ptr %13, ptr %2, align 8
  br label %19

14:                                               ; preds = %8
  %15 = icmp sgt i32 %6, -9
  br i1 %15, label %19, label %16

16:                                               ; preds = %14
  %17 = add nsw i32 %6, 16
  store i32 %17, ptr %5, align 8
  %18 = icmp samesign ult i32 %9, -7
  br i1 %18, label %22, label %19

19:                                               ; preds = %14, %16, %11
  %20 = load ptr, ptr %2, align 8
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 8
  store ptr %21, ptr %2, align 8
  br label %32

22:                                               ; preds = %16
  %23 = icmp sgt i32 %6, -17
  br i1 %23, label %32, label %24

24:                                               ; preds = %22
  %25 = add nsw i32 %6, 24
  store i32 %25, ptr %5, align 8
  %26 = icmp samesign ult i32 %17, -7
  br i1 %26, label %27, label %32

27:                                               ; preds = %24
  %28 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %29 = load ptr, ptr %28, align 8
  %30 = sext i32 %17 to i64
  %31 = getelementptr inbounds i8, ptr %29, i64 %30
  br label %35

32:                                               ; preds = %19, %24, %22
  %33 = load ptr, ptr %2, align 8
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 8
  store ptr %34, ptr %2, align 8
  br label %35

35:                                               ; preds = %32, %27
  %36 = phi ptr [ %31, %27 ], [ %33, %32 ]
  %37 = load i64, ptr %36, align 8, !tbaa !12
  store i64 %37, ptr @x, align 8, !tbaa !12
  %38 = trunc i64 %37 to i32
  call void @bar(i32 noundef %38)
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @f7(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.va_start.p0(ptr nonnull %2)
  store ptr %2, ptr @pap, align 8, !tbaa !14
  call void @bar(i32 noundef %0)
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @f8(i32 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #8
  call void @llvm.va_start.p0(ptr nonnull %2)
  store ptr %2, ptr @pap, align 8, !tbaa !14
  call void @bar(i32 noundef %0)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %4 = load i32, ptr %3, align 4
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %14, label %6

6:                                                ; preds = %1
  %7 = add nsw i32 %4, 16
  store i32 %7, ptr %3, align 4
  %8 = icmp samesign ult i32 %4, -15
  br i1 %8, label %9, label %14

9:                                                ; preds = %6
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %11 = load ptr, ptr %10, align 8
  %12 = sext i32 %4 to i64
  %13 = getelementptr inbounds i8, ptr %11, i64 %12
  br label %17

14:                                               ; preds = %6, %1
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store ptr %16, ptr %2, align 8
  br label %17

17:                                               ; preds = %14, %9
  %18 = phi ptr [ %13, %9 ], [ %15, %14 ]
  %19 = load double, ptr %18, align 8, !tbaa !10
  store double %19, ptr @d, align 8, !tbaa !10
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #8
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  store double 3.100000e+01, ptr @d, align 8, !tbaa !10
  tail call void (i32, ...) @f2(i32 poison, i64 noundef 28)
  %1 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %2 = icmp ne i32 %1, 28
  %3 = load i64, ptr @x, align 8
  %4 = icmp ne i64 %3, 28
  %5 = select i1 %2, i1 true, i1 %4
  br i1 %5, label %6, label %7

6:                                                ; preds = %0
  tail call void @abort() #7
  unreachable

7:                                                ; preds = %0
  tail call void (i32, ...) @f3(i32 poison, double noundef 1.310000e+02)
  %8 = load double, ptr @d, align 8, !tbaa !10
  %9 = fcmp une double %8, 1.310000e+02
  br i1 %9, label %10, label %11

10:                                               ; preds = %7
  tail call void @abort() #7
  unreachable

11:                                               ; preds = %7
  tail call void (i32, ...) @f4(i32 noundef 5, double noundef 1.600000e+01, i32 noundef 128)
  %12 = load i64, ptr @x, align 8, !tbaa !12
  %13 = icmp ne i64 %12, 16
  %14 = load i32, ptr @foo_arg, align 4
  %15 = icmp ne i32 %14, 128
  %16 = select i1 %13, i1 true, i1 %15
  br i1 %16, label %17, label %18

17:                                               ; preds = %11
  tail call void @abort() #7
  unreachable

18:                                               ; preds = %11
  tail call void (i32, ...) @f5(i32 noundef 16390, double noundef 1.700000e+01, i64 noundef 129)
  %19 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %20 = icmp eq i32 %19, 16390
  br i1 %20, label %22, label %21

21:                                               ; preds = %18
  tail call void @abort() #7
  unreachable

22:                                               ; preds = %18
  tail call void (i32, ...) @f6(i32 poison, i64 noundef 12, i64 noundef 14, i64 noundef -31)
  %23 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %24 = icmp eq i32 %23, -31
  br i1 %24, label %26, label %25

25:                                               ; preds = %22
  tail call void @abort() #7
  unreachable

26:                                               ; preds = %22
  tail call void (i32, ...) @f7(i32 noundef 16392, i64 noundef 14, fp128 noundef 0xL00000000000000004006060000000000, i32 noundef 17, double noundef 2.600000e+01)
  %27 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %28 = icmp eq i32 %27, 16392
  br i1 %28, label %30, label %29

29:                                               ; preds = %26
  tail call void @abort() #7
  unreachable

30:                                               ; preds = %26
  tail call void (i32, ...) @f8(i32 noundef 16392, i64 noundef 14, fp128 noundef 0xL00000000000000004006060000000000, i32 noundef 17, double noundef 2.700000e+01)
  %31 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %32 = icmp ne i32 %31, 16392
  %33 = load double, ptr @d, align 8
  %34 = fcmp une double %33, 2.700000e+01
  %35 = select i1 %32, i1 true, i1 %34
  br i1 %35, label %36, label %37

36:                                               ; preds = %30
  tail call void @abort() #7
  unreachable

37:                                               ; preds = %30
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #6 = { mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!13 = !{!"long", !8, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"p1 _ZTSSt9__va_list", !16, i64 0}
!16 = !{!"any pointer", !8, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"long long", !8, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"long double", !8, i64 0}
!21 = !{!16, !16, i64 0}
