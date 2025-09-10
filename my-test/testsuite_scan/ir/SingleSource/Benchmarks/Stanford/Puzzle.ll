; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Puzzle.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Stanford/Puzzle.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.element = type { i32, i32 }
%struct.complex = type { float, float }

@seed = dso_local local_unnamed_addr global i64 0, align 8
@piecemax = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 16
@p = dso_local local_unnamed_addr global [13 x [512 x i32]] zeroinitializer, align 16
@puzzl = dso_local local_unnamed_addr global [512 x i32] zeroinitializer, align 16
@piececount = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 16
@class = dso_local local_unnamed_addr global [13 x i32] zeroinitializer, align 16
@kount = dso_local local_unnamed_addr global i32 0, align 4
@n = dso_local local_unnamed_addr global i32 0, align 4
@.str.3 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@value = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@fixed = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@floated = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@permarray = dso_local local_unnamed_addr global [11 x i32] zeroinitializer, align 4
@pctr = dso_local local_unnamed_addr global i32 0, align 4
@tree = dso_local local_unnamed_addr global ptr null, align 8
@stack = dso_local local_unnamed_addr global [4 x i32] zeroinitializer, align 4
@cellspace = dso_local local_unnamed_addr global [19 x %struct.element] zeroinitializer, align 4
@freelist = dso_local local_unnamed_addr global i32 0, align 4
@movesdone = dso_local local_unnamed_addr global i32 0, align 4
@ima = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imb = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@imr = dso_local local_unnamed_addr global [41 x [41 x i32]] zeroinitializer, align 4
@rma = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmb = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@rmr = dso_local local_unnamed_addr global [41 x [41 x float]] zeroinitializer, align 4
@sortlist = dso_local local_unnamed_addr global [5001 x i32] zeroinitializer, align 4
@biggest = dso_local local_unnamed_addr global i32 0, align 4
@littlest = dso_local local_unnamed_addr global i32 0, align 4
@top = dso_local local_unnamed_addr global i32 0, align 4
@z = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@w = dso_local local_unnamed_addr global [257 x %struct.complex] zeroinitializer, align 4
@e = dso_local local_unnamed_addr global [130 x %struct.complex] zeroinitializer, align 4
@zr = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@zi = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@str = private unnamed_addr constant [17 x i8] c"Error1 in Puzzle\00", align 4
@str.4 = private unnamed_addr constant [18 x i8] c"Error2 in Puzzle.\00", align 4
@str.5 = private unnamed_addr constant [18 x i8] c"Error3 in Puzzle.\00", align 4

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

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @Fit(i32 noundef %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = sext i32 %0 to i64
  %4 = getelementptr inbounds i32, ptr @piecemax, i64 %3
  %5 = load i32, ptr %4, align 4, !tbaa !10
  %6 = icmp slt i32 %5, 0
  br i1 %6, label %25, label %7

7:                                                ; preds = %2
  %8 = getelementptr inbounds [512 x i32], ptr @p, i64 %3
  %9 = sext i32 %1 to i64
  %10 = add nuw i32 %5, 1
  %11 = zext i32 %10 to i64
  %12 = getelementptr i32, ptr @puzzl, i64 %9
  br label %13

13:                                               ; preds = %7, %22
  %14 = phi i64 [ 0, %7 ], [ %23, %22 ]
  %15 = getelementptr inbounds nuw i32, ptr %8, i64 %14
  %16 = load i32, ptr %15, align 4, !tbaa !10
  %17 = icmp eq i32 %16, 0
  br i1 %17, label %22, label %18

18:                                               ; preds = %13
  %19 = getelementptr i32, ptr %12, i64 %14
  %20 = load i32, ptr %19, align 4, !tbaa !10
  %21 = icmp eq i32 %20, 0
  br i1 %21, label %22, label %25

22:                                               ; preds = %13, %18
  %23 = add nuw nsw i64 %14, 1
  %24 = icmp eq i64 %23, %11
  br i1 %24, label %25, label %13, !llvm.loop !12

25:                                               ; preds = %18, %22, %2
  %26 = phi i32 [ 1, %2 ], [ 1, %22 ], [ 0, %18 ]
  ret i32 %26
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 -2147483648, 512) i32 @Place(i32 noundef %0, i32 noundef %1) local_unnamed_addr #3 {
  %3 = sext i32 %0 to i64
  %4 = getelementptr inbounds i32, ptr @piecemax, i64 %3
  %5 = load i32, ptr %4, align 4, !tbaa !10
  %6 = icmp slt i32 %5, 0
  br i1 %6, label %79, label %7

7:                                                ; preds = %2
  %8 = getelementptr inbounds [512 x i32], ptr @p, i64 %3
  %9 = sext i32 %1 to i64
  %10 = add nuw i32 %5, 1
  %11 = zext i32 %10 to i64
  %12 = getelementptr i32, ptr @puzzl, i64 %9
  %13 = icmp ult i32 %5, 7
  br i1 %13, label %67, label %14

14:                                               ; preds = %7
  %15 = and i64 %11, 4294967288
  br label %16

16:                                               ; preds = %62, %14
  %17 = phi i64 [ 0, %14 ], [ %63, %62 ]
  %18 = getelementptr inbounds nuw i32, ptr %8, i64 %17
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x i32>, ptr %18, align 4, !tbaa !10
  %21 = load <4 x i32>, ptr %19, align 4, !tbaa !10
  %22 = icmp ne <4 x i32> %20, zeroinitializer
  %23 = icmp ne <4 x i32> %21, zeroinitializer
  %24 = extractelement <4 x i1> %22, i64 0
  br i1 %24, label %25, label %27

25:                                               ; preds = %16
  %26 = getelementptr i32, ptr %12, i64 %17
  store i32 1, ptr %26, align 4, !tbaa !10
  br label %27

27:                                               ; preds = %25, %16
  %28 = extractelement <4 x i1> %22, i64 1
  br i1 %28, label %29, label %32

29:                                               ; preds = %27
  %30 = getelementptr i32, ptr %12, i64 %17
  %31 = getelementptr i8, ptr %30, i64 4
  store i32 1, ptr %31, align 4, !tbaa !10
  br label %32

32:                                               ; preds = %29, %27
  %33 = extractelement <4 x i1> %22, i64 2
  br i1 %33, label %34, label %37

34:                                               ; preds = %32
  %35 = getelementptr i32, ptr %12, i64 %17
  %36 = getelementptr i8, ptr %35, i64 8
  store i32 1, ptr %36, align 4, !tbaa !10
  br label %37

37:                                               ; preds = %34, %32
  %38 = extractelement <4 x i1> %22, i64 3
  br i1 %38, label %39, label %42

39:                                               ; preds = %37
  %40 = getelementptr i32, ptr %12, i64 %17
  %41 = getelementptr i8, ptr %40, i64 12
  store i32 1, ptr %41, align 4, !tbaa !10
  br label %42

42:                                               ; preds = %39, %37
  %43 = extractelement <4 x i1> %23, i64 0
  br i1 %43, label %44, label %47

44:                                               ; preds = %42
  %45 = getelementptr i32, ptr %12, i64 %17
  %46 = getelementptr i8, ptr %45, i64 16
  store i32 1, ptr %46, align 4, !tbaa !10
  br label %47

47:                                               ; preds = %44, %42
  %48 = extractelement <4 x i1> %23, i64 1
  br i1 %48, label %49, label %52

49:                                               ; preds = %47
  %50 = getelementptr i32, ptr %12, i64 %17
  %51 = getelementptr i8, ptr %50, i64 20
  store i32 1, ptr %51, align 4, !tbaa !10
  br label %52

52:                                               ; preds = %49, %47
  %53 = extractelement <4 x i1> %23, i64 2
  br i1 %53, label %54, label %57

54:                                               ; preds = %52
  %55 = getelementptr i32, ptr %12, i64 %17
  %56 = getelementptr i8, ptr %55, i64 24
  store i32 1, ptr %56, align 4, !tbaa !10
  br label %57

57:                                               ; preds = %54, %52
  %58 = extractelement <4 x i1> %23, i64 3
  br i1 %58, label %59, label %62

59:                                               ; preds = %57
  %60 = getelementptr i32, ptr %12, i64 %17
  %61 = getelementptr i8, ptr %60, i64 28
  store i32 1, ptr %61, align 4, !tbaa !10
  br label %62

62:                                               ; preds = %59, %57
  %63 = add nuw i64 %17, 8
  %64 = icmp eq i64 %63, %15
  br i1 %64, label %65, label %16, !llvm.loop !14

65:                                               ; preds = %62
  %66 = icmp eq i64 %15, %11
  br i1 %66, label %79, label %67

67:                                               ; preds = %7, %65
  %68 = phi i64 [ 0, %7 ], [ %15, %65 ]
  br label %69

69:                                               ; preds = %67, %76
  %70 = phi i64 [ %77, %76 ], [ %68, %67 ]
  %71 = getelementptr inbounds nuw i32, ptr %8, i64 %70
  %72 = load i32, ptr %71, align 4, !tbaa !10
  %73 = icmp eq i32 %72, 0
  br i1 %73, label %76, label %74

74:                                               ; preds = %69
  %75 = getelementptr i32, ptr %12, i64 %70
  store i32 1, ptr %75, align 4, !tbaa !10
  br label %76

76:                                               ; preds = %69, %74
  %77 = add nuw nsw i64 %70, 1
  %78 = icmp eq i64 %77, %11
  br i1 %78, label %79, label %69, !llvm.loop !17

79:                                               ; preds = %76, %65, %2
  %80 = getelementptr inbounds i32, ptr @class, i64 %3
  %81 = load i32, ptr %80, align 4, !tbaa !10
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds i32, ptr @piececount, i64 %82
  %84 = load i32, ptr %83, align 4, !tbaa !10
  %85 = add nsw i32 %84, -1
  store i32 %85, ptr %83, align 4, !tbaa !10
  %86 = icmp slt i32 %1, 512
  br i1 %86, label %87, label %100

87:                                               ; preds = %79
  %88 = sext i32 %1 to i64
  br label %89

89:                                               ; preds = %87, %94
  %90 = phi i64 [ %88, %87 ], [ %95, %94 ]
  %91 = getelementptr inbounds i32, ptr @puzzl, i64 %90
  %92 = load i32, ptr %91, align 4, !tbaa !10
  %93 = icmp eq i32 %92, 0
  br i1 %93, label %98, label %94

94:                                               ; preds = %89
  %95 = add nsw i64 %90, 1
  %96 = and i64 %95, 4294967295
  %97 = icmp eq i64 %96, 512
  br i1 %97, label %100, label %89, !llvm.loop !18

98:                                               ; preds = %89
  %99 = trunc nsw i64 %90 to i32
  br label %100

100:                                              ; preds = %94, %98, %79
  %101 = phi i32 [ 0, %79 ], [ %99, %98 ], [ 0, %94 ]
  ret i32 %101
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @Remove(i32 noundef %0, i32 noundef %1) local_unnamed_addr #3 {
  %3 = sext i32 %0 to i64
  %4 = getelementptr inbounds i32, ptr @piecemax, i64 %3
  %5 = load i32, ptr %4, align 4, !tbaa !10
  %6 = icmp slt i32 %5, 0
  br i1 %6, label %79, label %7

7:                                                ; preds = %2
  %8 = getelementptr inbounds [512 x i32], ptr @p, i64 %3
  %9 = sext i32 %1 to i64
  %10 = add nuw i32 %5, 1
  %11 = zext i32 %10 to i64
  %12 = getelementptr i32, ptr @puzzl, i64 %9
  %13 = icmp ult i32 %5, 7
  br i1 %13, label %67, label %14

14:                                               ; preds = %7
  %15 = and i64 %11, 4294967288
  br label %16

16:                                               ; preds = %62, %14
  %17 = phi i64 [ 0, %14 ], [ %63, %62 ]
  %18 = getelementptr inbounds nuw i32, ptr %8, i64 %17
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %20 = load <4 x i32>, ptr %18, align 4, !tbaa !10
  %21 = load <4 x i32>, ptr %19, align 4, !tbaa !10
  %22 = icmp ne <4 x i32> %20, zeroinitializer
  %23 = icmp ne <4 x i32> %21, zeroinitializer
  %24 = extractelement <4 x i1> %22, i64 0
  br i1 %24, label %25, label %27

25:                                               ; preds = %16
  %26 = getelementptr i32, ptr %12, i64 %17
  store i32 0, ptr %26, align 4, !tbaa !10
  br label %27

27:                                               ; preds = %25, %16
  %28 = extractelement <4 x i1> %22, i64 1
  br i1 %28, label %29, label %32

29:                                               ; preds = %27
  %30 = getelementptr i32, ptr %12, i64 %17
  %31 = getelementptr i8, ptr %30, i64 4
  store i32 0, ptr %31, align 4, !tbaa !10
  br label %32

32:                                               ; preds = %29, %27
  %33 = extractelement <4 x i1> %22, i64 2
  br i1 %33, label %34, label %37

34:                                               ; preds = %32
  %35 = getelementptr i32, ptr %12, i64 %17
  %36 = getelementptr i8, ptr %35, i64 8
  store i32 0, ptr %36, align 4, !tbaa !10
  br label %37

37:                                               ; preds = %34, %32
  %38 = extractelement <4 x i1> %22, i64 3
  br i1 %38, label %39, label %42

39:                                               ; preds = %37
  %40 = getelementptr i32, ptr %12, i64 %17
  %41 = getelementptr i8, ptr %40, i64 12
  store i32 0, ptr %41, align 4, !tbaa !10
  br label %42

42:                                               ; preds = %39, %37
  %43 = extractelement <4 x i1> %23, i64 0
  br i1 %43, label %44, label %47

44:                                               ; preds = %42
  %45 = getelementptr i32, ptr %12, i64 %17
  %46 = getelementptr i8, ptr %45, i64 16
  store i32 0, ptr %46, align 4, !tbaa !10
  br label %47

47:                                               ; preds = %44, %42
  %48 = extractelement <4 x i1> %23, i64 1
  br i1 %48, label %49, label %52

49:                                               ; preds = %47
  %50 = getelementptr i32, ptr %12, i64 %17
  %51 = getelementptr i8, ptr %50, i64 20
  store i32 0, ptr %51, align 4, !tbaa !10
  br label %52

52:                                               ; preds = %49, %47
  %53 = extractelement <4 x i1> %23, i64 2
  br i1 %53, label %54, label %57

54:                                               ; preds = %52
  %55 = getelementptr i32, ptr %12, i64 %17
  %56 = getelementptr i8, ptr %55, i64 24
  store i32 0, ptr %56, align 4, !tbaa !10
  br label %57

57:                                               ; preds = %54, %52
  %58 = extractelement <4 x i1> %23, i64 3
  br i1 %58, label %59, label %62

59:                                               ; preds = %57
  %60 = getelementptr i32, ptr %12, i64 %17
  %61 = getelementptr i8, ptr %60, i64 28
  store i32 0, ptr %61, align 4, !tbaa !10
  br label %62

62:                                               ; preds = %59, %57
  %63 = add nuw i64 %17, 8
  %64 = icmp eq i64 %63, %15
  br i1 %64, label %65, label %16, !llvm.loop !19

65:                                               ; preds = %62
  %66 = icmp eq i64 %15, %11
  br i1 %66, label %79, label %67

67:                                               ; preds = %7, %65
  %68 = phi i64 [ 0, %7 ], [ %15, %65 ]
  br label %69

69:                                               ; preds = %67, %76
  %70 = phi i64 [ %77, %76 ], [ %68, %67 ]
  %71 = getelementptr inbounds nuw i32, ptr %8, i64 %70
  %72 = load i32, ptr %71, align 4, !tbaa !10
  %73 = icmp eq i32 %72, 0
  br i1 %73, label %76, label %74

74:                                               ; preds = %69
  %75 = getelementptr i32, ptr %12, i64 %70
  store i32 0, ptr %75, align 4, !tbaa !10
  br label %76

76:                                               ; preds = %69, %74
  %77 = add nuw nsw i64 %70, 1
  %78 = icmp eq i64 %77, %11
  br i1 %78, label %79, label %69, !llvm.loop !20

79:                                               ; preds = %76, %65, %2
  %80 = getelementptr inbounds i32, ptr @class, i64 %3
  %81 = load i32, ptr %80, align 4, !tbaa !10
  %82 = sext i32 %81 to i64
  %83 = getelementptr inbounds i32, ptr @piececount, i64 %82
  %84 = load i32, ptr %83, align 4, !tbaa !10
  %85 = add nsw i32 %84, 1
  store i32 %85, ptr %83, align 4, !tbaa !10
  ret void
}

; Function Attrs: nofree nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @Trial(i32 noundef %0) local_unnamed_addr #4 {
  %2 = load i32, ptr @kount, align 4, !tbaa !10
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr @kount, align 4, !tbaa !10
  %4 = sext i32 %0 to i64
  %5 = getelementptr i32, ptr @puzzl, i64 %4
  %6 = icmp slt i32 %0, 512
  br label %7

7:                                                ; preds = %1, %200
  %8 = phi i64 [ 0, %1 ], [ %201, %200 ]
  %9 = getelementptr inbounds nuw i32, ptr @class, i64 %8
  %10 = load i32, ptr %9, align 4, !tbaa !10
  %11 = sext i32 %10 to i64
  %12 = getelementptr inbounds i32, ptr @piececount, i64 %11
  %13 = load i32, ptr %12, align 4, !tbaa !10
  %14 = icmp eq i32 %13, 0
  br i1 %14, label %200, label %15

15:                                               ; preds = %7
  %16 = getelementptr inbounds nuw i32, ptr @piecemax, i64 %8
  %17 = load i32, ptr %16, align 4, !tbaa !10
  %18 = icmp slt i32 %17, 0
  br i1 %18, label %102, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds nuw [512 x i32], ptr @p, i64 %8
  %21 = add nuw i32 %17, 1
  %22 = zext i32 %21 to i64
  br label %23

23:                                               ; preds = %32, %19
  %24 = phi i64 [ 0, %19 ], [ %33, %32 ]
  %25 = getelementptr inbounds nuw i32, ptr %20, i64 %24
  %26 = load i32, ptr %25, align 4, !tbaa !10
  %27 = icmp eq i32 %26, 0
  br i1 %27, label %32, label %28

28:                                               ; preds = %23
  %29 = getelementptr i32, ptr %5, i64 %24
  %30 = load i32, ptr %29, align 4, !tbaa !10
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %32, label %200

32:                                               ; preds = %28, %23
  %33 = add nuw nsw i64 %24, 1
  %34 = icmp eq i64 %33, %22
  br i1 %34, label %35, label %23, !llvm.loop !12

35:                                               ; preds = %32
  %36 = icmp ult i32 %17, 7
  br i1 %36, label %90, label %37

37:                                               ; preds = %35
  %38 = and i64 %22, 4294967288
  br label %39

39:                                               ; preds = %85, %37
  %40 = phi i64 [ 0, %37 ], [ %86, %85 ]
  %41 = getelementptr inbounds nuw i32, ptr %20, i64 %40
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 16
  %43 = load <4 x i32>, ptr %41, align 4, !tbaa !10
  %44 = load <4 x i32>, ptr %42, align 4, !tbaa !10
  %45 = icmp ne <4 x i32> %43, zeroinitializer
  %46 = icmp ne <4 x i32> %44, zeroinitializer
  %47 = extractelement <4 x i1> %45, i64 0
  br i1 %47, label %48, label %50

48:                                               ; preds = %39
  %49 = getelementptr i32, ptr %5, i64 %40
  store i32 1, ptr %49, align 4, !tbaa !10
  br label %50

50:                                               ; preds = %48, %39
  %51 = extractelement <4 x i1> %45, i64 1
  br i1 %51, label %52, label %55

52:                                               ; preds = %50
  %53 = getelementptr i32, ptr %5, i64 %40
  %54 = getelementptr i8, ptr %53, i64 4
  store i32 1, ptr %54, align 4, !tbaa !10
  br label %55

55:                                               ; preds = %52, %50
  %56 = extractelement <4 x i1> %45, i64 2
  br i1 %56, label %57, label %60

57:                                               ; preds = %55
  %58 = getelementptr i32, ptr %5, i64 %40
  %59 = getelementptr i8, ptr %58, i64 8
  store i32 1, ptr %59, align 4, !tbaa !10
  br label %60

60:                                               ; preds = %57, %55
  %61 = extractelement <4 x i1> %45, i64 3
  br i1 %61, label %62, label %65

62:                                               ; preds = %60
  %63 = getelementptr i32, ptr %5, i64 %40
  %64 = getelementptr i8, ptr %63, i64 12
  store i32 1, ptr %64, align 4, !tbaa !10
  br label %65

65:                                               ; preds = %62, %60
  %66 = extractelement <4 x i1> %46, i64 0
  br i1 %66, label %67, label %70

67:                                               ; preds = %65
  %68 = getelementptr i32, ptr %5, i64 %40
  %69 = getelementptr i8, ptr %68, i64 16
  store i32 1, ptr %69, align 4, !tbaa !10
  br label %70

70:                                               ; preds = %67, %65
  %71 = extractelement <4 x i1> %46, i64 1
  br i1 %71, label %72, label %75

72:                                               ; preds = %70
  %73 = getelementptr i32, ptr %5, i64 %40
  %74 = getelementptr i8, ptr %73, i64 20
  store i32 1, ptr %74, align 4, !tbaa !10
  br label %75

75:                                               ; preds = %72, %70
  %76 = extractelement <4 x i1> %46, i64 2
  br i1 %76, label %77, label %80

77:                                               ; preds = %75
  %78 = getelementptr i32, ptr %5, i64 %40
  %79 = getelementptr i8, ptr %78, i64 24
  store i32 1, ptr %79, align 4, !tbaa !10
  br label %80

80:                                               ; preds = %77, %75
  %81 = extractelement <4 x i1> %46, i64 3
  br i1 %81, label %82, label %85

82:                                               ; preds = %80
  %83 = getelementptr i32, ptr %5, i64 %40
  %84 = getelementptr i8, ptr %83, i64 28
  store i32 1, ptr %84, align 4, !tbaa !10
  br label %85

85:                                               ; preds = %82, %80
  %86 = add nuw i64 %40, 8
  %87 = icmp eq i64 %86, %38
  br i1 %87, label %88, label %39, !llvm.loop !21

88:                                               ; preds = %85
  %89 = icmp eq i64 %38, %22
  br i1 %89, label %102, label %90

90:                                               ; preds = %35, %88
  %91 = phi i64 [ 0, %35 ], [ %38, %88 ]
  br label %92

92:                                               ; preds = %90, %99
  %93 = phi i64 [ %100, %99 ], [ %91, %90 ]
  %94 = getelementptr inbounds nuw i32, ptr %20, i64 %93
  %95 = load i32, ptr %94, align 4, !tbaa !10
  %96 = icmp eq i32 %95, 0
  br i1 %96, label %99, label %97

97:                                               ; preds = %92
  %98 = getelementptr i32, ptr %5, i64 %93
  store i32 1, ptr %98, align 4, !tbaa !10
  br label %99

99:                                               ; preds = %97, %92
  %100 = add nuw nsw i64 %93, 1
  %101 = icmp eq i64 %100, %22
  br i1 %101, label %102, label %92, !llvm.loop !22

102:                                              ; preds = %99, %88, %15
  %103 = add nsw i32 %13, -1
  store i32 %103, ptr %12, align 4, !tbaa !10
  br i1 %6, label %104, label %115

104:                                              ; preds = %102, %109
  %105 = phi i64 [ %110, %109 ], [ %4, %102 ]
  %106 = getelementptr inbounds i32, ptr @puzzl, i64 %105
  %107 = load i32, ptr %106, align 4, !tbaa !10
  %108 = icmp eq i32 %107, 0
  br i1 %108, label %113, label %109

109:                                              ; preds = %104
  %110 = add nsw i64 %105, 1
  %111 = and i64 %110, 4294967295
  %112 = icmp eq i64 %111, 512
  br i1 %112, label %115, label %104, !llvm.loop !18

113:                                              ; preds = %104
  %114 = trunc nsw i64 %105 to i32
  br label %115

115:                                              ; preds = %109, %102, %113
  %116 = phi i32 [ 0, %102 ], [ %114, %113 ], [ 0, %109 ]
  %117 = tail call i32 @Trial(i32 noundef %116)
  %118 = icmp ne i32 %117, 0
  %119 = icmp eq i32 %116, 0
  %120 = or i1 %119, %118
  br i1 %120, label %203, label %121

121:                                              ; preds = %115
  %122 = load i32, ptr %16, align 4, !tbaa !10
  %123 = icmp slt i32 %122, 0
  br i1 %123, label %194, label %124

124:                                              ; preds = %121
  %125 = getelementptr inbounds nuw [512 x i32], ptr @p, i64 %8
  %126 = add nuw i32 %122, 1
  %127 = zext i32 %126 to i64
  %128 = icmp ult i32 %122, 7
  br i1 %128, label %182, label %129

129:                                              ; preds = %124
  %130 = and i64 %127, 4294967288
  br label %131

131:                                              ; preds = %177, %129
  %132 = phi i64 [ 0, %129 ], [ %178, %177 ]
  %133 = getelementptr inbounds nuw i32, ptr %125, i64 %132
  %134 = getelementptr inbounds nuw i8, ptr %133, i64 16
  %135 = load <4 x i32>, ptr %133, align 4, !tbaa !10
  %136 = load <4 x i32>, ptr %134, align 4, !tbaa !10
  %137 = icmp ne <4 x i32> %135, zeroinitializer
  %138 = icmp ne <4 x i32> %136, zeroinitializer
  %139 = extractelement <4 x i1> %137, i64 0
  br i1 %139, label %140, label %142

140:                                              ; preds = %131
  %141 = getelementptr i32, ptr %5, i64 %132
  store i32 0, ptr %141, align 4, !tbaa !10
  br label %142

142:                                              ; preds = %140, %131
  %143 = extractelement <4 x i1> %137, i64 1
  br i1 %143, label %144, label %147

144:                                              ; preds = %142
  %145 = getelementptr i32, ptr %5, i64 %132
  %146 = getelementptr i8, ptr %145, i64 4
  store i32 0, ptr %146, align 4, !tbaa !10
  br label %147

147:                                              ; preds = %144, %142
  %148 = extractelement <4 x i1> %137, i64 2
  br i1 %148, label %149, label %152

149:                                              ; preds = %147
  %150 = getelementptr i32, ptr %5, i64 %132
  %151 = getelementptr i8, ptr %150, i64 8
  store i32 0, ptr %151, align 4, !tbaa !10
  br label %152

152:                                              ; preds = %149, %147
  %153 = extractelement <4 x i1> %137, i64 3
  br i1 %153, label %154, label %157

154:                                              ; preds = %152
  %155 = getelementptr i32, ptr %5, i64 %132
  %156 = getelementptr i8, ptr %155, i64 12
  store i32 0, ptr %156, align 4, !tbaa !10
  br label %157

157:                                              ; preds = %154, %152
  %158 = extractelement <4 x i1> %138, i64 0
  br i1 %158, label %159, label %162

159:                                              ; preds = %157
  %160 = getelementptr i32, ptr %5, i64 %132
  %161 = getelementptr i8, ptr %160, i64 16
  store i32 0, ptr %161, align 4, !tbaa !10
  br label %162

162:                                              ; preds = %159, %157
  %163 = extractelement <4 x i1> %138, i64 1
  br i1 %163, label %164, label %167

164:                                              ; preds = %162
  %165 = getelementptr i32, ptr %5, i64 %132
  %166 = getelementptr i8, ptr %165, i64 20
  store i32 0, ptr %166, align 4, !tbaa !10
  br label %167

167:                                              ; preds = %164, %162
  %168 = extractelement <4 x i1> %138, i64 2
  br i1 %168, label %169, label %172

169:                                              ; preds = %167
  %170 = getelementptr i32, ptr %5, i64 %132
  %171 = getelementptr i8, ptr %170, i64 24
  store i32 0, ptr %171, align 4, !tbaa !10
  br label %172

172:                                              ; preds = %169, %167
  %173 = extractelement <4 x i1> %138, i64 3
  br i1 %173, label %174, label %177

174:                                              ; preds = %172
  %175 = getelementptr i32, ptr %5, i64 %132
  %176 = getelementptr i8, ptr %175, i64 28
  store i32 0, ptr %176, align 4, !tbaa !10
  br label %177

177:                                              ; preds = %174, %172
  %178 = add nuw i64 %132, 8
  %179 = icmp eq i64 %178, %130
  br i1 %179, label %180, label %131, !llvm.loop !23

180:                                              ; preds = %177
  %181 = icmp eq i64 %130, %127
  br i1 %181, label %194, label %182

182:                                              ; preds = %124, %180
  %183 = phi i64 [ 0, %124 ], [ %130, %180 ]
  br label %184

184:                                              ; preds = %182, %191
  %185 = phi i64 [ %192, %191 ], [ %183, %182 ]
  %186 = getelementptr inbounds nuw i32, ptr %125, i64 %185
  %187 = load i32, ptr %186, align 4, !tbaa !10
  %188 = icmp eq i32 %187, 0
  br i1 %188, label %191, label %189

189:                                              ; preds = %184
  %190 = getelementptr i32, ptr %5, i64 %185
  store i32 0, ptr %190, align 4, !tbaa !10
  br label %191

191:                                              ; preds = %189, %184
  %192 = add nuw nsw i64 %185, 1
  %193 = icmp eq i64 %192, %127
  br i1 %193, label %194, label %184, !llvm.loop !24

194:                                              ; preds = %191, %180, %121
  %195 = load i32, ptr %9, align 4, !tbaa !10
  %196 = sext i32 %195 to i64
  %197 = getelementptr inbounds i32, ptr @piececount, i64 %196
  %198 = load i32, ptr %197, align 4, !tbaa !10
  %199 = add nsw i32 %198, 1
  store i32 %199, ptr %197, align 4, !tbaa !10
  br label %200

200:                                              ; preds = %28, %7, %194
  %201 = add nuw nsw i64 %8, 1
  %202 = icmp eq i64 %201, 13
  br i1 %202, label %203, label %7, !llvm.loop !25

203:                                              ; preds = %200, %115
  %204 = phi i32 [ 1, %115 ], [ 0, %200 ]
  ret i32 %204
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @Puzzle() local_unnamed_addr #5 {
  store <4 x i32> splat (i32 1), ptr @puzzl, align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 16), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 32), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 48), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 64), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 80), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 96), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 112), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 128), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 144), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 160), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 176), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 192), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 208), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 224), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 240), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 256), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 272), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 288), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 304), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 320), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 336), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 352), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 368), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 384), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 400), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 416), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 432), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 448), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 464), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 480), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 496), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 512), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 528), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 544), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 560), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 576), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 592), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 608), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 624), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 640), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 656), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 672), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 688), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 704), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 720), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 736), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 752), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 768), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 784), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 800), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 816), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 832), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 848), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 864), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 880), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 896), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 912), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 928), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 944), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 960), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 976), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 992), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1008), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1024), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1040), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1056), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1072), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1088), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1104), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1120), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1136), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1152), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1168), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1184), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1200), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1216), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1232), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1248), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1264), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1280), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1296), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1312), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1328), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1344), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1360), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1376), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1392), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1408), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1424), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1440), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1456), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1472), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1488), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1504), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1520), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1536), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1552), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1568), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1584), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1600), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1616), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1632), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1648), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1664), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1680), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1696), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1712), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1728), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1744), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1760), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1776), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1792), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1808), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1824), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1840), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1856), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1872), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1888), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1904), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1920), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1936), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1952), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1968), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 1984), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 2000), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 2016), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 2032), align 16, !tbaa !10
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 292), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 324), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 356), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 388), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 420), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 548), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 580), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 612), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 644), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 676), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 804), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 836), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 868), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 900), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 932), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1060), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1092), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1124), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1156), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1188), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1316), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1348), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1380), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1412), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(20) getelementptr inbounds nuw (i8, ptr @puzzl, i64 1444), i8 0, i64 20, i1 false)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 16 dereferenceable(26608) getelementptr inbounds nuw (i8, ptr @p, i64 16), i8 0, i64 26608, i1 false), !tbaa !10
  store <4 x i32> splat (i32 1), ptr @p, align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 32), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 2048), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 2304), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 2560), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 2816), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4096), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4352), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4128), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4384), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4160), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4416), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4192), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4448), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 6144), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 6176), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 6208), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 6240), align 16, !tbaa !10
  store <4 x i32> zeroinitializer, ptr @class, align 16, !tbaa !10
  store <4 x i32> <i32 11, i32 193, i32 88, i32 25>, ptr @piecemax, align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 8192), align 16, !tbaa !10
  store <4 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 8448), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 10240), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 10496), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 10752), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 11008), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 10272), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 10528), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 10784), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 11040), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 12288), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 12296), align 8, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 14336), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 14368), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 14400), align 16, !tbaa !10
  store <4 x i32> <i32 0, i32 0, i32 1, i32 1>, ptr getelementptr inbounds nuw (i8, ptr @class, i64 16), align 16, !tbaa !10
  store <4 x i32> <i32 67, i32 200, i32 2, i32 16>, ptr getelementptr inbounds nuw (i8, ptr @piecemax, i64 16), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 16384), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 16640), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 16896), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 18432), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 18464), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 20480), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 20736), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 22528), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 22784), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 22560), align 16, !tbaa !10
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @p, i64 22816), align 16, !tbaa !10
  store <4 x i32> <i32 1, i32 2, i32 2, i32 2>, ptr getelementptr inbounds nuw (i8, ptr @class, i64 32), align 16, !tbaa !10
  store <4 x i32> <i32 128, i32 9, i32 65, i32 72>, ptr getelementptr inbounds nuw (i8, ptr @piecemax, i64 32), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 24576), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 24832), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 24608), align 16, !tbaa !10
  store <2 x i32> splat (i32 1), ptr getelementptr inbounds nuw (i8, ptr @p, i64 24864), align 16, !tbaa !10
  store i32 3, ptr getelementptr inbounds nuw (i8, ptr @class, i64 48), align 16, !tbaa !10
  store i32 73, ptr getelementptr inbounds nuw (i8, ptr @piecemax, i64 48), align 16, !tbaa !10
  store <4 x i32> <i32 13, i32 3, i32 1, i32 1>, ptr @piececount, align 16, !tbaa !10
  store i32 0, ptr @kount, align 4, !tbaa !10
  %1 = load i32, ptr @p, align 16, !tbaa !10
  %2 = icmp eq i32 %1, 0
  %3 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 292), align 4
  %4 = icmp eq i32 %3, 0
  %5 = select i1 %2, i1 true, i1 %4
  br i1 %5, label %6, label %128

6:                                                ; preds = %0
  %7 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 4), align 4, !tbaa !10
  %8 = icmp eq i32 %7, 0
  %9 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 296), align 8
  %10 = icmp eq i32 %9, 0
  %11 = select i1 %8, i1 true, i1 %10
  br i1 %11, label %12, label %128

12:                                               ; preds = %6
  %13 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 8), align 8, !tbaa !10
  %14 = icmp eq i32 %13, 0
  %15 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 300), align 4
  %16 = icmp eq i32 %15, 0
  %17 = select i1 %14, i1 true, i1 %16
  br i1 %17, label %18, label %128

18:                                               ; preds = %12
  %19 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 12), align 4, !tbaa !10
  %20 = icmp eq i32 %19, 0
  %21 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 304), align 16
  %22 = icmp eq i32 %21, 0
  %23 = select i1 %20, i1 true, i1 %22
  br i1 %23, label %24, label %128

24:                                               ; preds = %18
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 16), align 16, !tbaa !10
  %26 = icmp eq i32 %25, 0
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 308), align 4
  %28 = icmp eq i32 %27, 0
  %29 = select i1 %26, i1 true, i1 %28
  br i1 %29, label %30, label %128

30:                                               ; preds = %24
  %31 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 20), align 4, !tbaa !10
  %32 = icmp eq i32 %31, 0
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 312), align 8
  %34 = icmp eq i32 %33, 0
  %35 = select i1 %32, i1 true, i1 %34
  br i1 %35, label %36, label %128

36:                                               ; preds = %30
  %37 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 24), align 8, !tbaa !10
  %38 = icmp eq i32 %37, 0
  %39 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 316), align 4
  %40 = icmp eq i32 %39, 0
  %41 = select i1 %38, i1 true, i1 %40
  br i1 %41, label %42, label %128

42:                                               ; preds = %36
  %43 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 28), align 4, !tbaa !10
  %44 = icmp eq i32 %43, 0
  %45 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 320), align 16
  %46 = icmp eq i32 %45, 0
  %47 = select i1 %44, i1 true, i1 %46
  br i1 %47, label %48, label %128

48:                                               ; preds = %42
  %49 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 32), align 16, !tbaa !10
  %50 = icmp eq i32 %49, 0
  %51 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 324), align 4
  %52 = icmp eq i32 %51, 0
  %53 = select i1 %50, i1 true, i1 %52
  br i1 %53, label %54, label %128

54:                                               ; preds = %48
  %55 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 36), align 4, !tbaa !10
  %56 = icmp eq i32 %55, 0
  %57 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 328), align 8
  %58 = icmp eq i32 %57, 0
  %59 = select i1 %56, i1 true, i1 %58
  br i1 %59, label %60, label %128

60:                                               ; preds = %54
  %61 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 40), align 8, !tbaa !10
  %62 = icmp eq i32 %61, 0
  %63 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 332), align 4
  %64 = icmp eq i32 %63, 0
  %65 = select i1 %62, i1 true, i1 %64
  br i1 %65, label %66, label %128

66:                                               ; preds = %60
  %67 = load i32, ptr getelementptr inbounds nuw (i8, ptr @p, i64 44), align 4, !tbaa !10
  %68 = icmp eq i32 %67, 0
  %69 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 336), align 16
  %70 = icmp eq i32 %69, 0
  %71 = select i1 %68, i1 true, i1 %70
  br i1 %71, label %72, label %128

72:                                               ; preds = %66
  br i1 %2, label %74, label %73

73:                                               ; preds = %72
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 292), align 4, !tbaa !10
  br label %74

74:                                               ; preds = %73, %72
  br i1 %8, label %76, label %75

75:                                               ; preds = %74
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 296), align 8, !tbaa !10
  br label %76

76:                                               ; preds = %75, %74
  br i1 %14, label %78, label %77

77:                                               ; preds = %76
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 300), align 4, !tbaa !10
  br label %78

78:                                               ; preds = %77, %76
  br i1 %20, label %80, label %79

79:                                               ; preds = %78
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 304), align 16, !tbaa !10
  br label %80

80:                                               ; preds = %79, %78
  br i1 %26, label %82, label %81

81:                                               ; preds = %80
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 308), align 4, !tbaa !10
  br label %82

82:                                               ; preds = %81, %80
  br i1 %32, label %84, label %83

83:                                               ; preds = %82
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 312), align 8, !tbaa !10
  br label %84

84:                                               ; preds = %83, %82
  br i1 %38, label %86, label %85

85:                                               ; preds = %84
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 316), align 4, !tbaa !10
  br label %86

86:                                               ; preds = %85, %84
  br i1 %44, label %88, label %87

87:                                               ; preds = %86
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 320), align 16, !tbaa !10
  br label %88

88:                                               ; preds = %87, %86
  br i1 %50, label %90, label %89

89:                                               ; preds = %88
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 324), align 4, !tbaa !10
  br label %90

90:                                               ; preds = %89, %88
  br i1 %56, label %92, label %91

91:                                               ; preds = %90
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 328), align 8, !tbaa !10
  br label %92

92:                                               ; preds = %91, %90
  br i1 %62, label %94, label %93

93:                                               ; preds = %92
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 332), align 4, !tbaa !10
  br label %94

94:                                               ; preds = %93, %92
  br i1 %68, label %96, label %95

95:                                               ; preds = %94
  store i32 1, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 336), align 16, !tbaa !10
  br label %96

96:                                               ; preds = %95, %94
  store i32 12, ptr @piececount, align 16, !tbaa !10
  br label %97

97:                                               ; preds = %97, %96
  %98 = phi i64 [ 0, %96 ], [ %104, %97 ]
  %99 = getelementptr i32, ptr @puzzl, i64 %98
  %100 = getelementptr i8, ptr %99, i64 292
  %101 = load <4 x i32>, ptr %100, align 4, !tbaa !10
  %102 = freeze <4 x i32> %101
  %103 = icmp eq <4 x i32> %102, zeroinitializer
  %104 = add nuw i64 %98, 4
  %105 = bitcast <4 x i1> %103 to i4
  %106 = icmp ne i4 %105, 0
  %107 = icmp eq i64 %104, 436
  %108 = or i1 %106, %107
  br i1 %108, label %109, label %97, !llvm.loop !26

109:                                              ; preds = %97
  br i1 %106, label %110, label %114

110:                                              ; preds = %109
  %111 = tail call i64 @llvm.experimental.cttz.elts.i64.v4i1(<4 x i1> %103, i1 true)
  %112 = add i64 %98, %111
  %113 = add i64 %112, 73
  br label %123

114:                                              ; preds = %109
  %115 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 2036), align 4, !tbaa !10
  %116 = icmp eq i32 %115, 0
  br i1 %116, label %123, label %117

117:                                              ; preds = %114
  %118 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 2040), align 8, !tbaa !10
  %119 = icmp eq i32 %118, 0
  br i1 %119, label %123, label %120

120:                                              ; preds = %117
  %121 = load i32, ptr getelementptr inbounds nuw (i8, ptr @puzzl, i64 2044), align 4, !tbaa !10
  %122 = icmp eq i32 %121, 0
  br i1 %122, label %123, label %126

123:                                              ; preds = %114, %117, %120, %110
  %124 = phi i64 [ %113, %110 ], [ 509, %114 ], [ 510, %117 ], [ 511, %120 ]
  %125 = trunc nsw i64 %124 to i32
  br label %126

126:                                              ; preds = %120, %123
  %127 = phi i32 [ %125, %123 ], [ 0, %120 ]
  store i32 %127, ptr @n, align 4, !tbaa !10
  br label %131

128:                                              ; preds = %66, %60, %54, %48, %42, %36, %30, %24, %18, %12, %6, %0
  %129 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %130 = load i32, ptr @n, align 4, !tbaa !10
  br label %131

131:                                              ; preds = %128, %126
  %132 = phi i32 [ %130, %128 ], [ %127, %126 ]
  %133 = tail call i32 @Trial(i32 noundef %132)
  %134 = icmp eq i32 %133, 0
  br i1 %134, label %138, label %135

135:                                              ; preds = %131
  %136 = load i32, ptr @kount, align 4, !tbaa !10
  %137 = icmp eq i32 %136, 2005
  br i1 %137, label %141, label %138

138:                                              ; preds = %135, %131
  %139 = phi ptr [ @str.4, %131 ], [ @str.5, %135 ]
  %140 = tail call i32 @puts(ptr nonnull dereferenceable(1) %139)
  br label %141

141:                                              ; preds = %138, %135
  %142 = load i32, ptr @n, align 4, !tbaa !10
  %143 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %142)
  %144 = load i32, ptr @kount, align 4, !tbaa !10
  %145 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %144)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #6

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  tail call void @Puzzle()
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #7

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #8

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare i64 @llvm.experimental.cttz.elts.i64.v4i1(<4 x i1>, i1 immarg) #9

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree norecurse nosync nounwind memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree nounwind }
attributes #8 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #9 = { nocallback nofree nosync nounwind willreturn memory(none) }

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
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13, !15, !16}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !13, !16, !15}
!18 = distinct !{!18, !13}
!19 = distinct !{!19, !13, !15, !16}
!20 = distinct !{!20, !13, !16, !15}
!21 = distinct !{!21, !13, !15, !16}
!22 = distinct !{!22, !13, !16, !15}
!23 = distinct !{!23, !13, !15, !16}
!24 = distinct !{!24, !13, !16, !15}
!25 = distinct !{!25, !13}
!26 = distinct !{!26, !13, !15, !16}
