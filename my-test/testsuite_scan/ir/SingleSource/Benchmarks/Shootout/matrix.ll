; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/matrix.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/matrix.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [13 x i8] c"%d %d %d %d\0A\00", align 1

; Function Attrs: nofree nounwind memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @mkmatrix(i32 noundef %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = sext i32 %0 to i64
  %4 = shl nsw i64 %3, 3
  %5 = tail call noalias ptr @malloc(i64 noundef %4) #9
  %6 = icmp sgt i32 %0, 0
  br i1 %6, label %7, label %58

7:                                                ; preds = %2
  %8 = sext i32 %1 to i64
  %9 = shl nsw i64 %8, 2
  %10 = icmp sgt i32 %1, 0
  %11 = zext nneg i32 %0 to i64
  br i1 %10, label %12, label %52

12:                                               ; preds = %7
  %13 = zext nneg i32 %1 to i64
  %14 = icmp ult i32 %1, 8
  %15 = and i64 %13, 2147483640
  %16 = trunc nuw nsw i64 %15 to i32
  %17 = icmp eq i64 %15, %13
  br label %18

18:                                               ; preds = %12, %48
  %19 = phi i64 [ 0, %12 ], [ %50, %48 ]
  %20 = phi i32 [ 1, %12 ], [ %49, %48 ]
  %21 = tail call noalias ptr @malloc(i64 noundef %9) #9
  %22 = getelementptr inbounds nuw ptr, ptr %5, i64 %19
  store ptr %21, ptr %22, align 8, !tbaa !6
  br i1 %14, label %38, label %23

23:                                               ; preds = %18
  %24 = add i32 %20, %16
  %25 = insertelement <4 x i32> poison, i32 %20, i64 0
  %26 = shufflevector <4 x i32> %25, <4 x i32> poison, <4 x i32> zeroinitializer
  %27 = add <4 x i32> %26, <i32 0, i32 1, i32 2, i32 3>
  br label %28

28:                                               ; preds = %28, %23
  %29 = phi i64 [ 0, %23 ], [ %34, %28 ]
  %30 = phi <4 x i32> [ %27, %23 ], [ %35, %28 ]
  %31 = add <4 x i32> %30, splat (i32 4)
  %32 = getelementptr inbounds nuw i32, ptr %21, i64 %29
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 16
  store <4 x i32> %30, ptr %32, align 4, !tbaa !11
  store <4 x i32> %31, ptr %33, align 4, !tbaa !11
  %34 = add nuw i64 %29, 8
  %35 = add <4 x i32> %30, splat (i32 8)
  %36 = icmp eq i64 %34, %15
  br i1 %36, label %37, label %28, !llvm.loop !13

37:                                               ; preds = %28
  br i1 %17, label %48, label %38

38:                                               ; preds = %18, %37
  %39 = phi i64 [ 0, %18 ], [ %15, %37 ]
  %40 = phi i32 [ %20, %18 ], [ %24, %37 ]
  br label %41

41:                                               ; preds = %38, %41
  %42 = phi i64 [ %46, %41 ], [ %39, %38 ]
  %43 = phi i32 [ %44, %41 ], [ %40, %38 ]
  %44 = add nsw i32 %43, 1
  %45 = getelementptr inbounds nuw i32, ptr %21, i64 %42
  store i32 %43, ptr %45, align 4, !tbaa !11
  %46 = add nuw nsw i64 %42, 1
  %47 = icmp eq i64 %46, %13
  br i1 %47, label %48, label %41, !llvm.loop !17

48:                                               ; preds = %41, %37
  %49 = phi i32 [ %24, %37 ], [ %44, %41 ]
  %50 = add nuw nsw i64 %19, 1
  %51 = icmp eq i64 %50, %11
  br i1 %51, label %58, label %18, !llvm.loop !18

52:                                               ; preds = %7, %52
  %53 = phi i64 [ %56, %52 ], [ 0, %7 ]
  %54 = tail call noalias ptr @malloc(i64 noundef %9) #9
  %55 = getelementptr inbounds nuw ptr, ptr %5, i64 %53
  store ptr %54, ptr %55, align 8, !tbaa !6
  %56 = add nuw nsw i64 %53, 1
  %57 = icmp eq i64 %56, %11
  br i1 %57, label %58, label %52, !llvm.loop !18

58:                                               ; preds = %52, %48, %2
  ret ptr %5
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #1

; Function Attrs: nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @zeromatrix(i32 noundef %0, i32 noundef %1, ptr noundef readonly captures(none) %2) local_unnamed_addr #2 {
  %4 = icmp sgt i32 %0, 0
  %5 = icmp sgt i32 %1, 0
  %6 = and i1 %4, %5
  br i1 %6, label %7, label %17

7:                                                ; preds = %3
  %8 = zext nneg i32 %1 to i64
  %9 = shl nuw nsw i64 %8, 2
  %10 = zext nneg i32 %0 to i64
  br label %11

11:                                               ; preds = %7, %11
  %12 = phi i64 [ 0, %7 ], [ %15, %11 ]
  %13 = getelementptr inbounds nuw ptr, ptr %2, i64 %12
  %14 = load ptr, ptr %13, align 8, !tbaa !6
  tail call void @llvm.memset.p0.i64(ptr align 4 %14, i8 0, i64 %9, i1 false), !tbaa !11
  %15 = add nuw nsw i64 %12, 1
  %16 = icmp eq i64 %15, %10
  br i1 %16, label %17, label %11, !llvm.loop !19

17:                                               ; preds = %11, %3
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local void @freematrix(i32 noundef %0, ptr noundef captures(none) %1) local_unnamed_addr #3 {
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %12

4:                                                ; preds = %2
  %5 = zext nneg i32 %0 to i64
  br label %6

6:                                                ; preds = %4, %6
  %7 = phi i64 [ %5, %4 ], [ %8, %6 ]
  %8 = add nsw i64 %7, -1
  %9 = getelementptr inbounds nuw ptr, ptr %1, i64 %8
  %10 = load ptr, ptr %9, align 8, !tbaa !6
  tail call void @free(ptr noundef %10) #10
  %11 = icmp samesign ugt i64 %7, 1
  br i1 %11, label %6, label %12, !llvm.loop !20

12:                                               ; preds = %6, %2
  tail call void @free(ptr noundef %1) #10
  ret void
}

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #4

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local noundef ptr @mmult(i32 noundef %0, i32 noundef %1, ptr noundef readonly captures(none) %2, ptr noundef readonly captures(none) %3, ptr noundef readonly returned captures(ret: address, provenance) %4) local_unnamed_addr #5 {
  %6 = icmp sgt i32 %0, 0
  %7 = icmp sgt i32 %1, 0
  %8 = and i1 %6, %7
  br i1 %8, label %9, label %72

9:                                                ; preds = %5
  %10 = zext nneg i32 %0 to i64
  %11 = zext nneg i32 %1 to i64
  %12 = icmp ult i32 %1, 2
  %13 = and i64 %11, 2147483646
  %14 = icmp eq i64 %13, %11
  br label %15

15:                                               ; preds = %9, %69
  %16 = phi i64 [ 0, %9 ], [ %70, %69 ]
  %17 = getelementptr inbounds nuw ptr, ptr %2, i64 %16
  %18 = getelementptr inbounds nuw ptr, ptr %4, i64 %16
  %19 = load ptr, ptr %18, align 8, !tbaa !6
  %20 = load ptr, ptr %17, align 8, !tbaa !6
  br label %21

21:                                               ; preds = %64, %15
  %22 = phi i64 [ %67, %64 ], [ 0, %15 ]
  br i1 %12, label %48, label %23

23:                                               ; preds = %21, %23
  %24 = phi i64 [ %44, %23 ], [ 0, %21 ]
  %25 = phi i32 [ %42, %23 ], [ 0, %21 ]
  %26 = phi i32 [ %43, %23 ], [ 0, %21 ]
  %27 = or disjoint i64 %24, 1
  %28 = getelementptr inbounds nuw i32, ptr %20, i64 %24
  %29 = getelementptr inbounds nuw i32, ptr %20, i64 %27
  %30 = load i32, ptr %28, align 4, !tbaa !11
  %31 = load i32, ptr %29, align 4, !tbaa !11
  %32 = getelementptr inbounds nuw ptr, ptr %3, i64 %24
  %33 = getelementptr inbounds nuw ptr, ptr %3, i64 %27
  %34 = load ptr, ptr %32, align 8, !tbaa !6
  %35 = load ptr, ptr %33, align 8, !tbaa !6
  %36 = getelementptr inbounds nuw i32, ptr %34, i64 %22
  %37 = getelementptr inbounds nuw i32, ptr %35, i64 %22
  %38 = load i32, ptr %36, align 4, !tbaa !11
  %39 = load i32, ptr %37, align 4, !tbaa !11
  %40 = mul nsw i32 %38, %30
  %41 = mul nsw i32 %39, %31
  %42 = add i32 %40, %25
  %43 = add i32 %41, %26
  %44 = add nuw i64 %24, 2
  %45 = icmp eq i64 %44, %13
  br i1 %45, label %46, label %23, !llvm.loop !21

46:                                               ; preds = %23
  %47 = add i32 %43, %42
  br i1 %14, label %64, label %48

48:                                               ; preds = %21, %46
  %49 = phi i64 [ 0, %21 ], [ %13, %46 ]
  %50 = phi i32 [ 0, %21 ], [ %47, %46 ]
  br label %51

51:                                               ; preds = %48, %51
  %52 = phi i64 [ %62, %51 ], [ %49, %48 ]
  %53 = phi i32 [ %61, %51 ], [ %50, %48 ]
  %54 = getelementptr inbounds nuw i32, ptr %20, i64 %52
  %55 = load i32, ptr %54, align 4, !tbaa !11
  %56 = getelementptr inbounds nuw ptr, ptr %3, i64 %52
  %57 = load ptr, ptr %56, align 8, !tbaa !6
  %58 = getelementptr inbounds nuw i32, ptr %57, i64 %22
  %59 = load i32, ptr %58, align 4, !tbaa !11
  %60 = mul nsw i32 %59, %55
  %61 = add nsw i32 %60, %53
  %62 = add nuw nsw i64 %52, 1
  %63 = icmp eq i64 %62, %11
  br i1 %63, label %64, label %51, !llvm.loop !22

64:                                               ; preds = %51, %46
  %65 = phi i32 [ %47, %46 ], [ %61, %51 ]
  %66 = getelementptr inbounds nuw i32, ptr %19, i64 %22
  store i32 %65, ptr %66, align 4, !tbaa !11
  %67 = add nuw nsw i64 %22, 1
  %68 = icmp eq i64 %67, %11
  br i1 %68, label %69, label %21, !llvm.loop !23

69:                                               ; preds = %64
  %70 = add nuw nsw i64 %16, 1
  %71 = icmp eq i64 %70, %10
  br i1 %71, label %72, label %15, !llvm.loop !24

72:                                               ; preds = %69, %5
  ret ptr %4
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #3 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %9

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !25
  %7 = tail call i64 @strtol(ptr noundef nonnull captures(none) %6, ptr noundef null, i32 noundef 10) #10
  %8 = trunc i64 %7 to i32
  br label %9

9:                                                ; preds = %2, %4
  %10 = phi i32 [ %8, %4 ], [ 3000000, %2 ]
  %11 = tail call noalias dereferenceable_or_null(80) ptr @malloc(i64 noundef 80) #9
  %12 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  store ptr %12, ptr %11, align 8, !tbaa !6
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %12, align 4, !tbaa !11
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 16
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %13, align 4, !tbaa !11
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 32
  store <2 x i32> <i32 9, i32 10>, ptr %14, align 4, !tbaa !11
  %15 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %16 = getelementptr inbounds nuw i8, ptr %11, i64 8
  store ptr %15, ptr %16, align 8, !tbaa !6
  store <4 x i32> <i32 11, i32 12, i32 13, i32 14>, ptr %15, align 4, !tbaa !11
  %17 = getelementptr inbounds nuw i8, ptr %15, i64 16
  store <4 x i32> <i32 15, i32 16, i32 17, i32 18>, ptr %17, align 4, !tbaa !11
  %18 = getelementptr inbounds nuw i8, ptr %15, i64 32
  store <2 x i32> <i32 19, i32 20>, ptr %18, align 4, !tbaa !11
  %19 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %20 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store ptr %19, ptr %20, align 8, !tbaa !6
  store <4 x i32> <i32 21, i32 22, i32 23, i32 24>, ptr %19, align 4, !tbaa !11
  %21 = getelementptr inbounds nuw i8, ptr %19, i64 16
  store <4 x i32> <i32 25, i32 26, i32 27, i32 28>, ptr %21, align 4, !tbaa !11
  %22 = getelementptr inbounds nuw i8, ptr %19, i64 32
  store <2 x i32> <i32 29, i32 30>, ptr %22, align 4, !tbaa !11
  %23 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %24 = getelementptr inbounds nuw i8, ptr %11, i64 24
  store ptr %23, ptr %24, align 8, !tbaa !6
  store <4 x i32> <i32 31, i32 32, i32 33, i32 34>, ptr %23, align 4, !tbaa !11
  %25 = getelementptr inbounds nuw i8, ptr %23, i64 16
  store <4 x i32> <i32 35, i32 36, i32 37, i32 38>, ptr %25, align 4, !tbaa !11
  %26 = getelementptr inbounds nuw i8, ptr %23, i64 32
  store <2 x i32> <i32 39, i32 40>, ptr %26, align 4, !tbaa !11
  %27 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %28 = getelementptr inbounds nuw i8, ptr %11, i64 32
  store ptr %27, ptr %28, align 8, !tbaa !6
  store <4 x i32> <i32 41, i32 42, i32 43, i32 44>, ptr %27, align 4, !tbaa !11
  %29 = getelementptr inbounds nuw i8, ptr %27, i64 16
  store <4 x i32> <i32 45, i32 46, i32 47, i32 48>, ptr %29, align 4, !tbaa !11
  %30 = getelementptr inbounds nuw i8, ptr %27, i64 32
  store <2 x i32> <i32 49, i32 50>, ptr %30, align 4, !tbaa !11
  %31 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %32 = getelementptr inbounds nuw i8, ptr %11, i64 40
  store ptr %31, ptr %32, align 8, !tbaa !6
  store <4 x i32> <i32 51, i32 52, i32 53, i32 54>, ptr %31, align 4, !tbaa !11
  %33 = getelementptr inbounds nuw i8, ptr %31, i64 16
  store <4 x i32> <i32 55, i32 56, i32 57, i32 58>, ptr %33, align 4, !tbaa !11
  %34 = getelementptr inbounds nuw i8, ptr %31, i64 32
  store <2 x i32> <i32 59, i32 60>, ptr %34, align 4, !tbaa !11
  %35 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %36 = getelementptr inbounds nuw i8, ptr %11, i64 48
  store ptr %35, ptr %36, align 8, !tbaa !6
  store <4 x i32> <i32 61, i32 62, i32 63, i32 64>, ptr %35, align 4, !tbaa !11
  %37 = getelementptr inbounds nuw i8, ptr %35, i64 16
  store <4 x i32> <i32 65, i32 66, i32 67, i32 68>, ptr %37, align 4, !tbaa !11
  %38 = getelementptr inbounds nuw i8, ptr %35, i64 32
  store <2 x i32> <i32 69, i32 70>, ptr %38, align 4, !tbaa !11
  %39 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %40 = getelementptr inbounds nuw i8, ptr %11, i64 56
  store ptr %39, ptr %40, align 8, !tbaa !6
  store <4 x i32> <i32 71, i32 72, i32 73, i32 74>, ptr %39, align 4, !tbaa !11
  %41 = getelementptr inbounds nuw i8, ptr %39, i64 16
  store <4 x i32> <i32 75, i32 76, i32 77, i32 78>, ptr %41, align 4, !tbaa !11
  %42 = getelementptr inbounds nuw i8, ptr %39, i64 32
  store <2 x i32> <i32 79, i32 80>, ptr %42, align 4, !tbaa !11
  %43 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %44 = getelementptr inbounds nuw i8, ptr %11, i64 64
  store ptr %43, ptr %44, align 8, !tbaa !6
  store <4 x i32> <i32 81, i32 82, i32 83, i32 84>, ptr %43, align 4, !tbaa !11
  %45 = getelementptr inbounds nuw i8, ptr %43, i64 16
  store <4 x i32> <i32 85, i32 86, i32 87, i32 88>, ptr %45, align 4, !tbaa !11
  %46 = getelementptr inbounds nuw i8, ptr %43, i64 32
  store <2 x i32> <i32 89, i32 90>, ptr %46, align 4, !tbaa !11
  %47 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %48 = getelementptr inbounds nuw i8, ptr %11, i64 72
  store ptr %47, ptr %48, align 8, !tbaa !6
  store <4 x i32> <i32 91, i32 92, i32 93, i32 94>, ptr %47, align 4, !tbaa !11
  %49 = getelementptr inbounds nuw i8, ptr %47, i64 16
  store <4 x i32> <i32 95, i32 96, i32 97, i32 98>, ptr %49, align 4, !tbaa !11
  %50 = getelementptr inbounds nuw i8, ptr %47, i64 32
  store <2 x i32> <i32 99, i32 100>, ptr %50, align 4, !tbaa !11
  %51 = tail call noalias dereferenceable_or_null(80) ptr @malloc(i64 noundef 80) #9
  %52 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  store ptr %52, ptr %51, align 8, !tbaa !6
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %52, align 4, !tbaa !11
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 16
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %53, align 4, !tbaa !11
  %54 = getelementptr inbounds nuw i8, ptr %52, i64 32
  store <2 x i32> <i32 9, i32 10>, ptr %54, align 4, !tbaa !11
  %55 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %56 = getelementptr inbounds nuw i8, ptr %51, i64 8
  store ptr %55, ptr %56, align 8, !tbaa !6
  store <4 x i32> <i32 11, i32 12, i32 13, i32 14>, ptr %55, align 4, !tbaa !11
  %57 = getelementptr inbounds nuw i8, ptr %55, i64 16
  store <4 x i32> <i32 15, i32 16, i32 17, i32 18>, ptr %57, align 4, !tbaa !11
  %58 = getelementptr inbounds nuw i8, ptr %55, i64 32
  store <2 x i32> <i32 19, i32 20>, ptr %58, align 4, !tbaa !11
  %59 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %60 = getelementptr inbounds nuw i8, ptr %51, i64 16
  store ptr %59, ptr %60, align 8, !tbaa !6
  store <4 x i32> <i32 21, i32 22, i32 23, i32 24>, ptr %59, align 4, !tbaa !11
  %61 = getelementptr inbounds nuw i8, ptr %59, i64 16
  store <4 x i32> <i32 25, i32 26, i32 27, i32 28>, ptr %61, align 4, !tbaa !11
  %62 = getelementptr inbounds nuw i8, ptr %59, i64 32
  store <2 x i32> <i32 29, i32 30>, ptr %62, align 4, !tbaa !11
  %63 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %64 = getelementptr inbounds nuw i8, ptr %51, i64 24
  store ptr %63, ptr %64, align 8, !tbaa !6
  store <4 x i32> <i32 31, i32 32, i32 33, i32 34>, ptr %63, align 4, !tbaa !11
  %65 = getelementptr inbounds nuw i8, ptr %63, i64 16
  store <4 x i32> <i32 35, i32 36, i32 37, i32 38>, ptr %65, align 4, !tbaa !11
  %66 = getelementptr inbounds nuw i8, ptr %63, i64 32
  store <2 x i32> <i32 39, i32 40>, ptr %66, align 4, !tbaa !11
  %67 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %68 = getelementptr inbounds nuw i8, ptr %51, i64 32
  store ptr %67, ptr %68, align 8, !tbaa !6
  store <4 x i32> <i32 41, i32 42, i32 43, i32 44>, ptr %67, align 4, !tbaa !11
  %69 = getelementptr inbounds nuw i8, ptr %67, i64 16
  store <4 x i32> <i32 45, i32 46, i32 47, i32 48>, ptr %69, align 4, !tbaa !11
  %70 = getelementptr inbounds nuw i8, ptr %67, i64 32
  store <2 x i32> <i32 49, i32 50>, ptr %70, align 4, !tbaa !11
  %71 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %72 = getelementptr inbounds nuw i8, ptr %51, i64 40
  store ptr %71, ptr %72, align 8, !tbaa !6
  store <4 x i32> <i32 51, i32 52, i32 53, i32 54>, ptr %71, align 4, !tbaa !11
  %73 = getelementptr inbounds nuw i8, ptr %71, i64 16
  store <4 x i32> <i32 55, i32 56, i32 57, i32 58>, ptr %73, align 4, !tbaa !11
  %74 = getelementptr inbounds nuw i8, ptr %71, i64 32
  store <2 x i32> <i32 59, i32 60>, ptr %74, align 4, !tbaa !11
  %75 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %76 = getelementptr inbounds nuw i8, ptr %51, i64 48
  store ptr %75, ptr %76, align 8, !tbaa !6
  store <4 x i32> <i32 61, i32 62, i32 63, i32 64>, ptr %75, align 4, !tbaa !11
  %77 = getelementptr inbounds nuw i8, ptr %75, i64 16
  store <4 x i32> <i32 65, i32 66, i32 67, i32 68>, ptr %77, align 4, !tbaa !11
  %78 = getelementptr inbounds nuw i8, ptr %75, i64 32
  store <2 x i32> <i32 69, i32 70>, ptr %78, align 4, !tbaa !11
  %79 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %80 = getelementptr inbounds nuw i8, ptr %51, i64 56
  store ptr %79, ptr %80, align 8, !tbaa !6
  store <4 x i32> <i32 71, i32 72, i32 73, i32 74>, ptr %79, align 4, !tbaa !11
  %81 = getelementptr inbounds nuw i8, ptr %79, i64 16
  store <4 x i32> <i32 75, i32 76, i32 77, i32 78>, ptr %81, align 4, !tbaa !11
  %82 = getelementptr inbounds nuw i8, ptr %79, i64 32
  store <2 x i32> <i32 79, i32 80>, ptr %82, align 4, !tbaa !11
  %83 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %84 = getelementptr inbounds nuw i8, ptr %51, i64 64
  store ptr %83, ptr %84, align 8, !tbaa !6
  store <4 x i32> <i32 81, i32 82, i32 83, i32 84>, ptr %83, align 4, !tbaa !11
  %85 = getelementptr inbounds nuw i8, ptr %83, i64 16
  store <4 x i32> <i32 85, i32 86, i32 87, i32 88>, ptr %85, align 4, !tbaa !11
  %86 = getelementptr inbounds nuw i8, ptr %83, i64 32
  store <2 x i32> <i32 89, i32 90>, ptr %86, align 4, !tbaa !11
  %87 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %88 = getelementptr inbounds nuw i8, ptr %51, i64 72
  store ptr %87, ptr %88, align 8, !tbaa !6
  store <4 x i32> <i32 91, i32 92, i32 93, i32 94>, ptr %87, align 4, !tbaa !11
  %89 = getelementptr inbounds nuw i8, ptr %87, i64 16
  store <4 x i32> <i32 95, i32 96, i32 97, i32 98>, ptr %89, align 4, !tbaa !11
  %90 = getelementptr inbounds nuw i8, ptr %87, i64 32
  store <2 x i32> <i32 99, i32 100>, ptr %90, align 4, !tbaa !11
  %91 = tail call noalias dereferenceable_or_null(80) ptr @malloc(i64 noundef 80) #9
  %92 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  store ptr %92, ptr %91, align 8, !tbaa !6
  store <4 x i32> <i32 1, i32 2, i32 3, i32 4>, ptr %92, align 4, !tbaa !11
  %93 = getelementptr inbounds nuw i8, ptr %92, i64 16
  store <4 x i32> <i32 5, i32 6, i32 7, i32 8>, ptr %93, align 4, !tbaa !11
  %94 = getelementptr inbounds nuw i8, ptr %92, i64 32
  store <2 x i32> <i32 9, i32 10>, ptr %94, align 4, !tbaa !11
  %95 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %96 = getelementptr inbounds nuw i8, ptr %91, i64 8
  store ptr %95, ptr %96, align 8, !tbaa !6
  store <4 x i32> <i32 11, i32 12, i32 13, i32 14>, ptr %95, align 4, !tbaa !11
  %97 = getelementptr inbounds nuw i8, ptr %95, i64 16
  store <4 x i32> <i32 15, i32 16, i32 17, i32 18>, ptr %97, align 4, !tbaa !11
  %98 = getelementptr inbounds nuw i8, ptr %95, i64 32
  store <2 x i32> <i32 19, i32 20>, ptr %98, align 4, !tbaa !11
  %99 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %100 = getelementptr inbounds nuw i8, ptr %91, i64 16
  store ptr %99, ptr %100, align 8, !tbaa !6
  store <4 x i32> <i32 21, i32 22, i32 23, i32 24>, ptr %99, align 4, !tbaa !11
  %101 = getelementptr inbounds nuw i8, ptr %99, i64 16
  store <4 x i32> <i32 25, i32 26, i32 27, i32 28>, ptr %101, align 4, !tbaa !11
  %102 = getelementptr inbounds nuw i8, ptr %99, i64 32
  store <2 x i32> <i32 29, i32 30>, ptr %102, align 4, !tbaa !11
  %103 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %104 = getelementptr inbounds nuw i8, ptr %91, i64 24
  store ptr %103, ptr %104, align 8, !tbaa !6
  store <4 x i32> <i32 31, i32 32, i32 33, i32 34>, ptr %103, align 4, !tbaa !11
  %105 = getelementptr inbounds nuw i8, ptr %103, i64 16
  store <4 x i32> <i32 35, i32 36, i32 37, i32 38>, ptr %105, align 4, !tbaa !11
  %106 = getelementptr inbounds nuw i8, ptr %103, i64 32
  store <2 x i32> <i32 39, i32 40>, ptr %106, align 4, !tbaa !11
  %107 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %108 = getelementptr inbounds nuw i8, ptr %91, i64 32
  store ptr %107, ptr %108, align 8, !tbaa !6
  store <4 x i32> <i32 41, i32 42, i32 43, i32 44>, ptr %107, align 4, !tbaa !11
  %109 = getelementptr inbounds nuw i8, ptr %107, i64 16
  store <4 x i32> <i32 45, i32 46, i32 47, i32 48>, ptr %109, align 4, !tbaa !11
  %110 = getelementptr inbounds nuw i8, ptr %107, i64 32
  store <2 x i32> <i32 49, i32 50>, ptr %110, align 4, !tbaa !11
  %111 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %112 = getelementptr inbounds nuw i8, ptr %91, i64 40
  store ptr %111, ptr %112, align 8, !tbaa !6
  store <4 x i32> <i32 51, i32 52, i32 53, i32 54>, ptr %111, align 4, !tbaa !11
  %113 = getelementptr inbounds nuw i8, ptr %111, i64 16
  store <4 x i32> <i32 55, i32 56, i32 57, i32 58>, ptr %113, align 4, !tbaa !11
  %114 = getelementptr inbounds nuw i8, ptr %111, i64 32
  store <2 x i32> <i32 59, i32 60>, ptr %114, align 4, !tbaa !11
  %115 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %116 = getelementptr inbounds nuw i8, ptr %91, i64 48
  store ptr %115, ptr %116, align 8, !tbaa !6
  store <4 x i32> <i32 61, i32 62, i32 63, i32 64>, ptr %115, align 4, !tbaa !11
  %117 = getelementptr inbounds nuw i8, ptr %115, i64 16
  store <4 x i32> <i32 65, i32 66, i32 67, i32 68>, ptr %117, align 4, !tbaa !11
  %118 = getelementptr inbounds nuw i8, ptr %115, i64 32
  store <2 x i32> <i32 69, i32 70>, ptr %118, align 4, !tbaa !11
  %119 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %120 = getelementptr inbounds nuw i8, ptr %91, i64 56
  store ptr %119, ptr %120, align 8, !tbaa !6
  store <4 x i32> <i32 71, i32 72, i32 73, i32 74>, ptr %119, align 4, !tbaa !11
  %121 = getelementptr inbounds nuw i8, ptr %119, i64 16
  store <4 x i32> <i32 75, i32 76, i32 77, i32 78>, ptr %121, align 4, !tbaa !11
  %122 = getelementptr inbounds nuw i8, ptr %119, i64 32
  store <2 x i32> <i32 79, i32 80>, ptr %122, align 4, !tbaa !11
  %123 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %124 = getelementptr inbounds nuw i8, ptr %91, i64 64
  store ptr %123, ptr %124, align 8, !tbaa !6
  store <4 x i32> <i32 81, i32 82, i32 83, i32 84>, ptr %123, align 4, !tbaa !11
  %125 = getelementptr inbounds nuw i8, ptr %123, i64 16
  store <4 x i32> <i32 85, i32 86, i32 87, i32 88>, ptr %125, align 4, !tbaa !11
  %126 = getelementptr inbounds nuw i8, ptr %123, i64 32
  store <2 x i32> <i32 89, i32 90>, ptr %126, align 4, !tbaa !11
  %127 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #9
  %128 = getelementptr inbounds nuw i8, ptr %91, i64 72
  store ptr %127, ptr %128, align 8, !tbaa !6
  store <4 x i32> <i32 91, i32 92, i32 93, i32 94>, ptr %127, align 4, !tbaa !11
  %129 = getelementptr inbounds nuw i8, ptr %127, i64 16
  store <4 x i32> <i32 95, i32 96, i32 97, i32 98>, ptr %129, align 4, !tbaa !11
  %130 = getelementptr inbounds nuw i8, ptr %127, i64 32
  store <2 x i32> <i32 99, i32 100>, ptr %130, align 4, !tbaa !11
  %131 = icmp sgt i32 %10, 0
  br i1 %131, label %132, label %220

132:                                              ; preds = %9
  %133 = load ptr, ptr %51, align 8, !tbaa !6
  %134 = load ptr, ptr %56, align 8, !tbaa !6
  %135 = load ptr, ptr %60, align 8, !tbaa !6
  %136 = load ptr, ptr %64, align 8, !tbaa !6
  %137 = load ptr, ptr %68, align 8, !tbaa !6
  %138 = load ptr, ptr %72, align 8, !tbaa !6
  %139 = load ptr, ptr %76, align 8, !tbaa !6
  %140 = load ptr, ptr %80, align 8, !tbaa !6
  %141 = load ptr, ptr %84, align 8, !tbaa !6
  %142 = load ptr, ptr %88, align 8, !tbaa !6
  br label %143

143:                                              ; preds = %132, %217
  %144 = phi i32 [ %218, %217 ], [ 0, %132 ]
  br label %145

145:                                              ; preds = %143, %214
  %146 = phi i64 [ %215, %214 ], [ 0, %143 ]
  %147 = getelementptr inbounds nuw ptr, ptr %11, i64 %146
  %148 = getelementptr inbounds nuw ptr, ptr %91, i64 %146
  %149 = load ptr, ptr %148, align 8, !tbaa !6
  %150 = load ptr, ptr %147, align 8, !tbaa !6
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 4
  %152 = getelementptr inbounds nuw i8, ptr %150, i64 8
  %153 = getelementptr inbounds nuw i8, ptr %150, i64 12
  %154 = getelementptr inbounds nuw i8, ptr %150, i64 16
  %155 = getelementptr inbounds nuw i8, ptr %150, i64 20
  %156 = getelementptr inbounds nuw i8, ptr %150, i64 24
  %157 = getelementptr inbounds nuw i8, ptr %150, i64 28
  %158 = getelementptr inbounds nuw i8, ptr %150, i64 32
  %159 = getelementptr inbounds nuw i8, ptr %150, i64 36
  br label %160

160:                                              ; preds = %160, %145
  %161 = phi i64 [ %212, %160 ], [ 0, %145 ]
  %162 = load i32, ptr %150, align 4, !tbaa !11
  %163 = getelementptr inbounds nuw i32, ptr %133, i64 %161
  %164 = load i32, ptr %163, align 4, !tbaa !11
  %165 = mul nsw i32 %164, %162
  %166 = load i32, ptr %151, align 4, !tbaa !11
  %167 = getelementptr inbounds nuw i32, ptr %134, i64 %161
  %168 = load i32, ptr %167, align 4, !tbaa !11
  %169 = mul nsw i32 %168, %166
  %170 = add nsw i32 %169, %165
  %171 = load i32, ptr %152, align 4, !tbaa !11
  %172 = getelementptr inbounds nuw i32, ptr %135, i64 %161
  %173 = load i32, ptr %172, align 4, !tbaa !11
  %174 = mul nsw i32 %173, %171
  %175 = add nsw i32 %174, %170
  %176 = load i32, ptr %153, align 4, !tbaa !11
  %177 = getelementptr inbounds nuw i32, ptr %136, i64 %161
  %178 = load i32, ptr %177, align 4, !tbaa !11
  %179 = mul nsw i32 %178, %176
  %180 = add nsw i32 %179, %175
  %181 = load i32, ptr %154, align 4, !tbaa !11
  %182 = getelementptr inbounds nuw i32, ptr %137, i64 %161
  %183 = load i32, ptr %182, align 4, !tbaa !11
  %184 = mul nsw i32 %183, %181
  %185 = add nsw i32 %184, %180
  %186 = load i32, ptr %155, align 4, !tbaa !11
  %187 = getelementptr inbounds nuw i32, ptr %138, i64 %161
  %188 = load i32, ptr %187, align 4, !tbaa !11
  %189 = mul nsw i32 %188, %186
  %190 = add nsw i32 %189, %185
  %191 = load i32, ptr %156, align 4, !tbaa !11
  %192 = getelementptr inbounds nuw i32, ptr %139, i64 %161
  %193 = load i32, ptr %192, align 4, !tbaa !11
  %194 = mul nsw i32 %193, %191
  %195 = add nsw i32 %194, %190
  %196 = load i32, ptr %157, align 4, !tbaa !11
  %197 = getelementptr inbounds nuw i32, ptr %140, i64 %161
  %198 = load i32, ptr %197, align 4, !tbaa !11
  %199 = mul nsw i32 %198, %196
  %200 = add nsw i32 %199, %195
  %201 = load i32, ptr %158, align 4, !tbaa !11
  %202 = getelementptr inbounds nuw i32, ptr %141, i64 %161
  %203 = load i32, ptr %202, align 4, !tbaa !11
  %204 = mul nsw i32 %203, %201
  %205 = add nsw i32 %204, %200
  %206 = load i32, ptr %159, align 4, !tbaa !11
  %207 = getelementptr inbounds nuw i32, ptr %142, i64 %161
  %208 = load i32, ptr %207, align 4, !tbaa !11
  %209 = mul nsw i32 %208, %206
  %210 = add nsw i32 %209, %205
  %211 = getelementptr inbounds nuw i32, ptr %149, i64 %161
  store i32 %210, ptr %211, align 4, !tbaa !11
  %212 = add nuw nsw i64 %161, 1
  %213 = icmp eq i64 %212, 10
  br i1 %213, label %214, label %160, !llvm.loop !23

214:                                              ; preds = %160
  %215 = add nuw nsw i64 %146, 1
  %216 = icmp eq i64 %215, 10
  br i1 %216, label %217, label %145, !llvm.loop !24

217:                                              ; preds = %214
  %218 = add nuw nsw i32 %144, 1
  %219 = icmp eq i32 %218, %10
  br i1 %219, label %220, label %143, !llvm.loop !27

220:                                              ; preds = %217, %9
  %221 = load ptr, ptr %91, align 8, !tbaa !6
  %222 = load i32, ptr %221, align 4, !tbaa !11
  %223 = load ptr, ptr %100, align 8, !tbaa !6
  %224 = getelementptr inbounds nuw i8, ptr %223, i64 12
  %225 = load i32, ptr %224, align 4, !tbaa !11
  %226 = load ptr, ptr %104, align 8, !tbaa !6
  %227 = getelementptr inbounds nuw i8, ptr %226, i64 8
  %228 = load i32, ptr %227, align 4, !tbaa !11
  %229 = load ptr, ptr %108, align 8, !tbaa !6
  %230 = getelementptr inbounds nuw i8, ptr %229, i64 16
  %231 = load i32, ptr %230, align 4, !tbaa !11
  %232 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %222, i32 noundef %225, i32 noundef %228, i32 noundef %231)
  %233 = load ptr, ptr %48, align 8, !tbaa !6
  tail call void @free(ptr noundef %233) #10
  %234 = load ptr, ptr %44, align 8, !tbaa !6
  tail call void @free(ptr noundef %234) #10
  %235 = load ptr, ptr %40, align 8, !tbaa !6
  tail call void @free(ptr noundef %235) #10
  %236 = load ptr, ptr %36, align 8, !tbaa !6
  tail call void @free(ptr noundef %236) #10
  %237 = load ptr, ptr %32, align 8, !tbaa !6
  tail call void @free(ptr noundef %237) #10
  %238 = load ptr, ptr %28, align 8, !tbaa !6
  tail call void @free(ptr noundef %238) #10
  %239 = load ptr, ptr %24, align 8, !tbaa !6
  tail call void @free(ptr noundef %239) #10
  %240 = load ptr, ptr %20, align 8, !tbaa !6
  tail call void @free(ptr noundef %240) #10
  %241 = load ptr, ptr %16, align 8, !tbaa !6
  tail call void @free(ptr noundef %241) #10
  %242 = load ptr, ptr %11, align 8, !tbaa !6
  tail call void @free(ptr noundef %242) #10
  tail call void @free(ptr noundef nonnull %11) #10
  %243 = load ptr, ptr %88, align 8, !tbaa !6
  tail call void @free(ptr noundef %243) #10
  %244 = load ptr, ptr %84, align 8, !tbaa !6
  tail call void @free(ptr noundef %244) #10
  %245 = load ptr, ptr %80, align 8, !tbaa !6
  tail call void @free(ptr noundef %245) #10
  %246 = load ptr, ptr %76, align 8, !tbaa !6
  tail call void @free(ptr noundef %246) #10
  %247 = load ptr, ptr %72, align 8, !tbaa !6
  tail call void @free(ptr noundef %247) #10
  %248 = load ptr, ptr %68, align 8, !tbaa !6
  tail call void @free(ptr noundef %248) #10
  %249 = load ptr, ptr %64, align 8, !tbaa !6
  tail call void @free(ptr noundef %249) #10
  %250 = load ptr, ptr %60, align 8, !tbaa !6
  tail call void @free(ptr noundef %250) #10
  %251 = load ptr, ptr %56, align 8, !tbaa !6
  tail call void @free(ptr noundef %251) #10
  %252 = load ptr, ptr %51, align 8, !tbaa !6
  tail call void @free(ptr noundef %252) #10
  tail call void @free(ptr noundef nonnull %51) #10
  tail call void @free(ptr noundef %127) #10
  tail call void @free(ptr noundef %123) #10
  tail call void @free(ptr noundef %119) #10
  tail call void @free(ptr noundef %115) #10
  %253 = load ptr, ptr %112, align 8, !tbaa !6
  tail call void @free(ptr noundef %253) #10
  tail call void @free(ptr noundef %229) #10
  tail call void @free(ptr noundef %226) #10
  tail call void @free(ptr noundef %223) #10
  %254 = load ptr, ptr %96, align 8, !tbaa !6
  tail call void @free(ptr noundef %254) #10
  tail call void @free(ptr noundef %221) #10
  tail call void @free(ptr noundef nonnull %91) #10
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #6

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #7

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #8

attributes #0 = { nofree nounwind memory(write, argmem: none, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #9 = { nounwind allocsize(0) }
attributes #10 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 int", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = distinct !{!13, !14, !15, !16}
!14 = !{!"llvm.loop.mustprogress"}
!15 = !{!"llvm.loop.isvectorized", i32 1}
!16 = !{!"llvm.loop.unroll.runtime.disable"}
!17 = distinct !{!17, !14, !16, !15}
!18 = distinct !{!18, !14}
!19 = distinct !{!19, !14}
!20 = distinct !{!20, !14}
!21 = distinct !{!21, !14, !15, !16}
!22 = distinct !{!22, !14, !15}
!23 = distinct !{!23, !14}
!24 = distinct !{!24, !14}
!25 = !{!26, !26, i64 0}
!26 = !{!"p1 omnipotent char", !8, i64 0}
!27 = distinct !{!27, !14}
