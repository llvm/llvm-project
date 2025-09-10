; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Towers.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Towers.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.element = type { i32, i32 }
%struct.complex = type { float, float }

@seed = dso_local local_unnamed_addr global i64 0, align 8
@.str = private unnamed_addr constant [22 x i8] c" Error in Towers: %s\0A\00", align 1
@stack = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 4
@freelist = dso_local local_unnamed_addr global i32 0, align 4
@cellspace = dso_local local_unnamed_addr global [19 x %struct.element] zeroinitializer, align 4
@.str.1 = private unnamed_addr constant [16 x i8] c"out of space   \00", align 1
@.str.2 = private unnamed_addr constant [16 x i8] c"disc size error\00", align 1
@.str.3 = private unnamed_addr constant [16 x i8] c"nothing to pop \00", align 1
@movesdone = dso_local local_unnamed_addr global i32 0, align 4
@.str.5 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@value = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@fixed = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@floated = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@permarray = dso_local local_unnamed_addr global [11 x i32] zeroinitializer, align 4
@pctr = dso_local local_unnamed_addr global i32 0, align 4
@tree = dso_local local_unnamed_addr global ptr null, align 8
@ima = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imb = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imr = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@rma = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmb = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmr = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@piececount = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 4
@class = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 4
@piecemax = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 4
@puzzl = dso_local local_unnamed_addr global [512 x i32] zeroinitializer, align 4
@p = dso_local local_unnamed_addr global [13 x [512 x i32]] zeroinitializer, align 4
@n = dso_local local_unnamed_addr global i32 0, align 4
@kount = dso_local local_unnamed_addr global i32 0, align 4
@sortlist = dso_local local_unnamed_addr global [5001 x i32] zeroinitializer, align 4
@biggest = dso_local local_unnamed_addr global i32 0, align 4
@littlest = dso_local local_unnamed_addr global i32 0, align 4
@top = dso_local local_unnamed_addr global i32 0, align 4
@z = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@w = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@e = dso_local local_unnamed_addr global [130 x %struct.complex] zeroinitializer, align 4
@zr = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@zi = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@str = private unnamed_addr constant [18 x i8] c" Error in Towers.\00", align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @Initrand() local_unnamed_addr #0 {
  store i64 74755, ptr @seed, align 8, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 65536) i32 @Rand() local_unnamed_addr #1 {
  %1 = load i64, ptr @seed, align 8, !tbaa !6
  %2 = mul nsw i64 %1, 1309
  %3 = add nsw i64 %2, 13849
  %4 = and i64 %3, 65535
  store i64 %4, ptr @seed, align 8, !tbaa !6
  %5 = trunc nuw nsw i64 %4 to i32
  ret i32 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @Error(ptr noundef %0) local_unnamed_addr #2 {
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef %0)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @Makenull(i32 noundef %0) local_unnamed_addr #0 {
  %2 = sext i32 %0 to i64
  %3 = getelementptr inbounds i32, ptr @stack, i64 %2
  store i32 0, ptr %3, align 4, !tbaa !10
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local range(i32 0, -2147483648) i32 @Getelement() local_unnamed_addr #2 {
  %1 = load i32, ptr @freelist, align 4, !tbaa !10
  %2 = icmp sgt i32 %1, 0
  br i1 %2, label %3, label %7

3:                                                ; preds = %0
  %4 = zext nneg i32 %1 to i64
  %5 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %4, i32 1
  %6 = load i32, ptr %5, align 4, !tbaa !12
  store i32 %6, ptr @freelist, align 4, !tbaa !10
  br label %9

7:                                                ; preds = %0
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1)
  br label %9

9:                                                ; preds = %7, %3
  %10 = phi i32 [ %1, %3 ], [ 0, %7 ]
  ret i32 %10
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @Push(i32 noundef %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = sext i32 %1 to i64
  %4 = getelementptr inbounds i32, ptr @stack, i64 %3
  %5 = load i32, ptr %4, align 4, !tbaa !10
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %14

7:                                                ; preds = %2
  %8 = zext nneg i32 %5 to i64
  %9 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %8
  %10 = load i32, ptr %9, align 4, !tbaa !14
  %11 = icmp sgt i32 %10, %0
  br i1 %11, label %14, label %12

12:                                               ; preds = %7
  %13 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.2)
  br label %30

14:                                               ; preds = %2, %7
  %15 = load i32, ptr @freelist, align 4, !tbaa !10
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %17, label %21

17:                                               ; preds = %14
  %18 = zext nneg i32 %15 to i64
  %19 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %18, i32 1
  %20 = load i32, ptr %19, align 4, !tbaa !12
  store i32 %20, ptr @freelist, align 4, !tbaa !10
  br label %24

21:                                               ; preds = %14
  %22 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1)
  %23 = load i32, ptr %4, align 4, !tbaa !10
  br label %24

24:                                               ; preds = %17, %21
  %25 = phi i32 [ %5, %17 ], [ %23, %21 ]
  %26 = phi i32 [ %15, %17 ], [ 0, %21 ]
  %27 = zext nneg i32 %26 to i64
  %28 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %27
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 4
  store i32 %25, ptr %29, align 4, !tbaa !12
  store i32 %26, ptr %4, align 4, !tbaa !10
  store i32 %0, ptr %28, align 4, !tbaa !14
  br label %30

30:                                               ; preds = %12, %24
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @Init(i32 noundef %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = sext i32 %0 to i64
  %4 = getelementptr inbounds i32, ptr @stack, i64 %3
  store i32 0, ptr %4, align 4, !tbaa !10
  %5 = icmp sgt i32 %1, 0
  br i1 %5, label %9, label %38

6:                                                ; preds = %36
  %7 = add nsw i32 %11, -1
  %8 = load i32, ptr %4, align 4, !tbaa !10
  br label %9

9:                                                ; preds = %2, %6
  %10 = phi i32 [ %8, %6 ], [ 0, %2 ]
  %11 = phi i32 [ %7, %6 ], [ %1, %2 ]
  %12 = icmp sgt i32 %10, 0
  br i1 %12, label %13, label %20

13:                                               ; preds = %9
  %14 = zext nneg i32 %10 to i64
  %15 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %14
  %16 = load i32, ptr %15, align 4, !tbaa !14
  %17 = icmp sgt i32 %16, %11
  br i1 %17, label %20, label %18

18:                                               ; preds = %13
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.2)
  br label %36

20:                                               ; preds = %13, %9
  %21 = load i32, ptr @freelist, align 4, !tbaa !10
  %22 = icmp sgt i32 %21, 0
  br i1 %22, label %23, label %27

23:                                               ; preds = %20
  %24 = zext nneg i32 %21 to i64
  %25 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %24, i32 1
  %26 = load i32, ptr %25, align 4, !tbaa !12
  store i32 %26, ptr @freelist, align 4, !tbaa !10
  br label %30

27:                                               ; preds = %20
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1)
  %29 = load i32, ptr %4, align 4, !tbaa !10
  br label %30

30:                                               ; preds = %27, %23
  %31 = phi i32 [ %10, %23 ], [ %29, %27 ]
  %32 = phi i32 [ %21, %23 ], [ 0, %27 ]
  %33 = zext nneg i32 %32 to i64
  %34 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %33
  %35 = getelementptr inbounds nuw i8, ptr %34, i64 4
  store i32 %31, ptr %35, align 4, !tbaa !12
  store i32 %32, ptr %4, align 4, !tbaa !10
  store i32 %11, ptr %34, align 4, !tbaa !14
  br label %36

36:                                               ; preds = %18, %30
  %37 = icmp sgt i32 %11, 1
  br i1 %37, label %6, label %38, !llvm.loop !15

38:                                               ; preds = %36, %2
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @Pop(i32 noundef %0) local_unnamed_addr #2 {
  %2 = sext i32 %0 to i64
  %3 = getelementptr inbounds i32, ptr @stack, i64 %2
  %4 = load i32, ptr %3, align 4, !tbaa !10
  %5 = icmp sgt i32 %4, 0
  br i1 %5, label %6, label %13

6:                                                ; preds = %1
  %7 = zext nneg i32 %4 to i64
  %8 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %7
  %9 = load i32, ptr %8, align 4, !tbaa !14
  %10 = getelementptr inbounds nuw i8, ptr %8, i64 4
  %11 = load i32, ptr %10, align 4, !tbaa !12
  %12 = load i32, ptr @freelist, align 4, !tbaa !10
  store i32 %12, ptr %10, align 4, !tbaa !12
  store i32 %4, ptr @freelist, align 4, !tbaa !10
  store i32 %11, ptr %3, align 4, !tbaa !10
  br label %15

13:                                               ; preds = %1
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.3)
  br label %15

15:                                               ; preds = %13, %6
  %16 = phi i32 [ %9, %6 ], [ 0, %13 ]
  ret i32 %16
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @Move(i32 noundef %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = sext i32 %0 to i64
  %4 = getelementptr inbounds i32, ptr @stack, i64 %3
  %5 = load i32, ptr %4, align 4, !tbaa !10
  %6 = icmp sgt i32 %5, 0
  br i1 %6, label %7, label %14

7:                                                ; preds = %2
  %8 = zext nneg i32 %5 to i64
  %9 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %8
  %10 = load i32, ptr %9, align 4, !tbaa !14
  %11 = getelementptr inbounds nuw i8, ptr %9, i64 4
  %12 = load i32, ptr %11, align 4, !tbaa !12
  %13 = load i32, ptr @freelist, align 4, !tbaa !10
  store i32 %13, ptr %11, align 4, !tbaa !12
  store i32 %5, ptr @freelist, align 4, !tbaa !10
  store i32 %12, ptr %4, align 4, !tbaa !10
  br label %16

14:                                               ; preds = %2
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.3)
  br label %16

16:                                               ; preds = %7, %14
  %17 = phi i32 [ %10, %7 ], [ 0, %14 ]
  %18 = sext i32 %1 to i64
  %19 = getelementptr inbounds i32, ptr @stack, i64 %18
  %20 = load i32, ptr %19, align 4, !tbaa !10
  %21 = icmp sgt i32 %20, 0
  br i1 %21, label %22, label %29

22:                                               ; preds = %16
  %23 = zext nneg i32 %20 to i64
  %24 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %23
  %25 = load i32, ptr %24, align 4, !tbaa !14
  %26 = icmp sgt i32 %25, %17
  br i1 %26, label %29, label %27

27:                                               ; preds = %22
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.2)
  br label %45

29:                                               ; preds = %22, %16
  %30 = load i32, ptr @freelist, align 4, !tbaa !10
  %31 = icmp sgt i32 %30, 0
  br i1 %31, label %32, label %36

32:                                               ; preds = %29
  %33 = zext nneg i32 %30 to i64
  %34 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %33, i32 1
  %35 = load i32, ptr %34, align 4, !tbaa !12
  store i32 %35, ptr @freelist, align 4, !tbaa !10
  br label %39

36:                                               ; preds = %29
  %37 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1)
  %38 = load i32, ptr %19, align 4, !tbaa !10
  br label %39

39:                                               ; preds = %36, %32
  %40 = phi i32 [ %20, %32 ], [ %38, %36 ]
  %41 = phi i32 [ %30, %32 ], [ 0, %36 ]
  %42 = zext nneg i32 %41 to i64
  %43 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %42
  %44 = getelementptr inbounds nuw i8, ptr %43, i64 4
  store i32 %40, ptr %44, align 4, !tbaa !12
  store i32 %41, ptr %19, align 4, !tbaa !10
  store i32 %17, ptr %43, align 4, !tbaa !14
  br label %45

45:                                               ; preds = %27, %39
  %46 = load i32, ptr @movesdone, align 4, !tbaa !10
  %47 = add nsw i32 %46, 1
  store i32 %47, ptr @movesdone, align 4, !tbaa !10
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @tower(i32 noundef %0, i32 noundef %1, i32 noundef %2) local_unnamed_addr #2 {
  %4 = icmp eq i32 %2, 1
  br i1 %4, label %8, label %5

5:                                                ; preds = %3
  %6 = sext i32 %1 to i64
  %7 = getelementptr inbounds i32, ptr @stack, i64 %6
  br label %55

8:                                                ; preds = %101, %3
  %9 = phi i32 [ %0, %3 ], [ %59, %101 ]
  %10 = sext i32 %9 to i64
  %11 = getelementptr inbounds i32, ptr @stack, i64 %10
  %12 = load i32, ptr %11, align 4, !tbaa !10
  %13 = icmp sgt i32 %12, 0
  br i1 %13, label %14, label %21

14:                                               ; preds = %8
  %15 = zext nneg i32 %12 to i64
  %16 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %15
  %17 = load i32, ptr %16, align 4, !tbaa !14
  %18 = getelementptr inbounds nuw i8, ptr %16, i64 4
  %19 = load i32, ptr %18, align 4, !tbaa !12
  %20 = load i32, ptr @freelist, align 4, !tbaa !10
  store i32 %20, ptr %18, align 4, !tbaa !12
  store i32 %12, ptr @freelist, align 4, !tbaa !10
  store i32 %19, ptr %11, align 4, !tbaa !10
  br label %23

21:                                               ; preds = %8
  %22 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.3)
  br label %23

23:                                               ; preds = %21, %14
  %24 = phi i32 [ %17, %14 ], [ 0, %21 ]
  %25 = sext i32 %1 to i64
  %26 = getelementptr inbounds i32, ptr @stack, i64 %25
  %27 = load i32, ptr %26, align 4, !tbaa !10
  %28 = icmp sgt i32 %27, 0
  br i1 %28, label %29, label %36

29:                                               ; preds = %23
  %30 = zext nneg i32 %27 to i64
  %31 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %30
  %32 = load i32, ptr %31, align 4, !tbaa !14
  %33 = icmp sgt i32 %32, %24
  br i1 %33, label %36, label %34

34:                                               ; preds = %29
  %35 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.2)
  br label %52

36:                                               ; preds = %29, %23
  %37 = load i32, ptr @freelist, align 4, !tbaa !10
  %38 = icmp sgt i32 %37, 0
  br i1 %38, label %39, label %43

39:                                               ; preds = %36
  %40 = zext nneg i32 %37 to i64
  %41 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %40, i32 1
  %42 = load i32, ptr %41, align 4, !tbaa !12
  store i32 %42, ptr @freelist, align 4, !tbaa !10
  br label %46

43:                                               ; preds = %36
  %44 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1)
  %45 = load i32, ptr %26, align 4, !tbaa !10
  br label %46

46:                                               ; preds = %43, %39
  %47 = phi i32 [ %27, %39 ], [ %45, %43 ]
  %48 = phi i32 [ %37, %39 ], [ 0, %43 ]
  %49 = zext nneg i32 %48 to i64
  %50 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %49
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 4
  store i32 %47, ptr %51, align 4, !tbaa !12
  store i32 %48, ptr %26, align 4, !tbaa !10
  store i32 %24, ptr %50, align 4, !tbaa !14
  br label %52

52:                                               ; preds = %34, %46
  %53 = load i32, ptr @movesdone, align 4, !tbaa !10
  %54 = add nsw i32 %53, 1
  store i32 %54, ptr @movesdone, align 4, !tbaa !10
  ret void

55:                                               ; preds = %5, %101
  %56 = phi i32 [ %2, %5 ], [ %60, %101 ]
  %57 = phi i32 [ %0, %5 ], [ %59, %101 ]
  %58 = add i32 %1, %57
  %59 = sub i32 6, %58
  %60 = add nsw i32 %56, -1
  tail call void @tower(i32 noundef %57, i32 noundef %59, i32 noundef %60)
  %61 = sext i32 %57 to i64
  %62 = getelementptr inbounds i32, ptr @stack, i64 %61
  %63 = load i32, ptr %62, align 4, !tbaa !10
  %64 = icmp sgt i32 %63, 0
  br i1 %64, label %65, label %72

65:                                               ; preds = %55
  %66 = zext nneg i32 %63 to i64
  %67 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %66
  %68 = load i32, ptr %67, align 4, !tbaa !14
  %69 = getelementptr inbounds nuw i8, ptr %67, i64 4
  %70 = load i32, ptr %69, align 4, !tbaa !12
  %71 = load i32, ptr @freelist, align 4, !tbaa !10
  store i32 %71, ptr %69, align 4, !tbaa !12
  store i32 %63, ptr @freelist, align 4, !tbaa !10
  store i32 %70, ptr %62, align 4, !tbaa !10
  br label %74

72:                                               ; preds = %55
  %73 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.3)
  br label %74

74:                                               ; preds = %72, %65
  %75 = phi i32 [ %68, %65 ], [ 0, %72 ]
  %76 = load i32, ptr %7, align 4, !tbaa !10
  %77 = icmp sgt i32 %76, 0
  br i1 %77, label %78, label %85

78:                                               ; preds = %74
  %79 = zext nneg i32 %76 to i64
  %80 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %79
  %81 = load i32, ptr %80, align 4, !tbaa !14
  %82 = icmp sgt i32 %81, %75
  br i1 %82, label %85, label %83

83:                                               ; preds = %78
  %84 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.2)
  br label %101

85:                                               ; preds = %78, %74
  %86 = load i32, ptr @freelist, align 4, !tbaa !10
  %87 = icmp sgt i32 %86, 0
  br i1 %87, label %88, label %92

88:                                               ; preds = %85
  %89 = zext nneg i32 %86 to i64
  %90 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %89, i32 1
  %91 = load i32, ptr %90, align 4, !tbaa !12
  store i32 %91, ptr @freelist, align 4, !tbaa !10
  br label %95

92:                                               ; preds = %85
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1)
  %94 = load i32, ptr %7, align 4, !tbaa !10
  br label %95

95:                                               ; preds = %92, %88
  %96 = phi i32 [ %76, %88 ], [ %94, %92 ]
  %97 = phi i32 [ %86, %88 ], [ 0, %92 ]
  %98 = zext nneg i32 %97 to i64
  %99 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %98
  %100 = getelementptr inbounds nuw i8, ptr %99, i64 4
  store i32 %96, ptr %100, align 4, !tbaa !12
  store i32 %97, ptr %7, align 4, !tbaa !10
  store i32 %75, ptr %99, align 4, !tbaa !14
  br label %101

101:                                              ; preds = %83, %95
  %102 = load i32, ptr @movesdone, align 4, !tbaa !10
  %103 = add nsw i32 %102, 1
  store i32 %103, ptr @movesdone, align 4, !tbaa !10
  %104 = icmp eq i32 %60, 1
  br i1 %104, label %8, label %55
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @Towers() local_unnamed_addr #2 {
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 12), align 4, !tbaa !12
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 20), align 4, !tbaa !12
  store i32 2, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 28), align 4, !tbaa !12
  store i32 3, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 36), align 4, !tbaa !12
  store i32 4, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 44), align 4, !tbaa !12
  store i32 5, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 52), align 4, !tbaa !12
  store i32 6, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 60), align 4, !tbaa !12
  store i32 7, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 68), align 4, !tbaa !12
  store i32 8, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 76), align 4, !tbaa !12
  store i32 9, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 84), align 4, !tbaa !12
  store i32 10, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 92), align 4, !tbaa !12
  store i32 11, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 100), align 4, !tbaa !12
  store i32 12, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 108), align 4, !tbaa !12
  store i32 13, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 116), align 4, !tbaa !12
  store i32 14, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 124), align 4, !tbaa !12
  store i32 15, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 132), align 4, !tbaa !12
  store i32 16, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 140), align 4, !tbaa !12
  store i32 17, ptr getelementptr inbounds nuw (i8, ptr @cellspace, i64 148), align 4, !tbaa !12
  store i32 18, ptr @freelist, align 4, !tbaa !10
  store i32 0, ptr getelementptr inbounds nuw (i8, ptr @stack, i64 4), align 4, !tbaa !10
  br label %4

1:                                                ; preds = %31
  %2 = add nsw i32 %6, -1
  %3 = load i32, ptr getelementptr inbounds nuw (i8, ptr @stack, i64 4), align 4, !tbaa !10
  br label %4

4:                                                ; preds = %1, %0
  %5 = phi i32 [ %3, %1 ], [ 0, %0 ]
  %6 = phi i32 [ %2, %1 ], [ 14, %0 ]
  %7 = icmp sgt i32 %5, 0
  br i1 %7, label %8, label %15

8:                                                ; preds = %4
  %9 = zext nneg i32 %5 to i64
  %10 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %9
  %11 = load i32, ptr %10, align 4, !tbaa !14
  %12 = icmp sgt i32 %11, %6
  br i1 %12, label %15, label %13

13:                                               ; preds = %8
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.2)
  br label %31

15:                                               ; preds = %8, %4
  %16 = load i32, ptr @freelist, align 4, !tbaa !10
  %17 = icmp sgt i32 %16, 0
  br i1 %17, label %18, label %22

18:                                               ; preds = %15
  %19 = zext nneg i32 %16 to i64
  %20 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %19, i32 1
  %21 = load i32, ptr %20, align 4, !tbaa !12
  store i32 %21, ptr @freelist, align 4, !tbaa !10
  br label %25

22:                                               ; preds = %15
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull @.str.1)
  %24 = load i32, ptr getelementptr inbounds nuw (i8, ptr @stack, i64 4), align 4, !tbaa !10
  br label %25

25:                                               ; preds = %22, %18
  %26 = phi i32 [ %5, %18 ], [ %24, %22 ]
  %27 = phi i32 [ %16, %18 ], [ 0, %22 ]
  %28 = zext nneg i32 %27 to i64
  %29 = getelementptr inbounds nuw %struct.element, ptr @cellspace, i64 %28
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 4
  store i32 %26, ptr %30, align 4, !tbaa !12
  store i32 %27, ptr getelementptr inbounds nuw (i8, ptr @stack, i64 4), align 4, !tbaa !10
  store i32 %6, ptr %29, align 4, !tbaa !14
  br label %31

31:                                               ; preds = %25, %13
  %32 = icmp samesign ugt i32 %6, 1
  br i1 %32, label %1, label %33, !llvm.loop !15

33:                                               ; preds = %31
  store <2 x i32> zeroinitializer, ptr getelementptr inbounds nuw (i8, ptr @stack, i64 8), align 4, !tbaa !10
  store i32 0, ptr @movesdone, align 4, !tbaa !10
  tail call void @tower(i32 noundef 1, i32 noundef 2, i32 noundef 14)
  %34 = load i32, ptr @movesdone, align 4, !tbaa !10
  %35 = icmp eq i32 %34, 16383
  br i1 %35, label %39, label %36

36:                                               ; preds = %33
  %37 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %38 = load i32, ptr @movesdone, align 4, !tbaa !10
  br label %39

39:                                               ; preds = %36, %33
  %40 = phi i32 [ %38, %36 ], [ 16383, %33 ]
  %41 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %40)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  tail call void @Towers()
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = !{!13, !11, i64 4}
!13 = !{!"element", !11, i64 0, !11, i64 4}
!14 = !{!13, !11, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
