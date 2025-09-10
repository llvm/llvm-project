; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/lists.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/lists.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.DLL = type { i32, ptr, ptr }

@.str = private unnamed_addr constant [12 x i8] c"length: %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [28 x i8] c"i:%3d  v:%3d  n:%3d  p:%3d\0A\00", align 1
@.str.3 = private unnamed_addr constant [31 x i8] c"[val of next of tail is:  %d]\0A\00", align 1
@.str.7 = private unnamed_addr constant [42 x i8] c"li1 first value wrong, wanted %d, got %d\0A\00", align 1
@.str.8 = private unnamed_addr constant [37 x i8] c"last value wrong, wanted %d, got %d\0A\00", align 1
@.str.9 = private unnamed_addr constant [42 x i8] c"li2 first value wrong, wanted %d, got %d\0A\00", align 1
@.str.12 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@str = private unnamed_addr constant [33 x i8] c"[last entry points to list head]\00", align 4
@str.13 = private unnamed_addr constant [26 x i8] c"li2 and li1 are not equal\00", align 4
@str.16 = private unnamed_addr constant [26 x i8] c"li1 and li2 are not equal\00", align 4

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @list_push_tail(ptr noundef %0, ptr noundef %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %4 = load ptr, ptr %3, align 8, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr %1, ptr %5, align 8, !tbaa !13
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %0, ptr %6, align 8, !tbaa !13
  store ptr %1, ptr %3, align 8, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store ptr %4, ptr %7, align 8, !tbaa !6
  %8 = load i32, ptr %0, align 8, !tbaa !14
  %9 = add nsw i32 %8, 1
  store i32 %9, ptr %0, align 8, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define dso_local ptr @list_pop_tail(ptr noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr %0, align 8, !tbaa !14
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %11, label %4

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %6 = load ptr, ptr %5, align 8, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %8 = load ptr, ptr %7, align 8, !tbaa !6
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store ptr %0, ptr %9, align 8, !tbaa !13
  store ptr %8, ptr %5, align 8, !tbaa !6
  %10 = add nsw i32 %2, -1
  store i32 %10, ptr %0, align 8, !tbaa !14
  br label %11

11:                                               ; preds = %1, %4
  %12 = phi ptr [ %6, %4 ], [ null, %1 ]
  ret ptr %12
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @list_push_head(ptr noundef %0, ptr noundef %1) local_unnamed_addr #0 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = load ptr, ptr %3, align 8, !tbaa !13
  store ptr %1, ptr %3, align 8, !tbaa !13
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store ptr %1, ptr %5, align 8, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %4, ptr %6, align 8, !tbaa !13
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store ptr %0, ptr %7, align 8, !tbaa !6
  %8 = load i32, ptr %0, align 8, !tbaa !14
  %9 = add nsw i32 %8, 1
  store i32 %9, ptr %0, align 8, !tbaa !14
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable
define dso_local ptr @list_pop_head(ptr noundef %0) local_unnamed_addr #1 {
  %2 = load i32, ptr %0, align 8, !tbaa !14
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %11, label %4

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !13
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !13
  store ptr %8, ptr %5, align 8, !tbaa !13
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store ptr %0, ptr %9, align 8, !tbaa !6
  %10 = add nsw i32 %2, -1
  store i32 %10, ptr %0, align 8, !tbaa !14
  br label %11

11:                                               ; preds = %1, %4
  %12 = phi ptr [ %6, %4 ], [ null, %1 ]
  ret ptr %12
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @list_equal(ptr noundef readonly captures(address) %0, ptr noundef readonly captures(address) %1) local_unnamed_addr #2 {
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %4 = load ptr, ptr %3, align 8, !tbaa !13
  %5 = icmp eq ptr %4, %0
  br i1 %5, label %19, label %6

6:                                                ; preds = %2, %13
  %7 = phi ptr [ %17, %13 ], [ %4, %2 ]
  %8 = phi ptr [ %15, %13 ], [ %1, %2 ]
  %9 = phi ptr [ %7, %13 ], [ %0, %2 ]
  %10 = load i32, ptr %9, align 8, !tbaa !14
  %11 = load i32, ptr %8, align 8, !tbaa !14
  %12 = icmp eq i32 %10, %11
  br i1 %12, label %13, label %30

13:                                               ; preds = %6
  %14 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %15 = load ptr, ptr %14, align 8, !tbaa !13
  %16 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %17 = load ptr, ptr %16, align 8, !tbaa !13
  %18 = icmp eq ptr %17, %0
  br i1 %18, label %19, label %6, !llvm.loop !15

19:                                               ; preds = %13, %2
  %20 = phi ptr [ %0, %2 ], [ %7, %13 ]
  %21 = phi ptr [ %1, %2 ], [ %15, %13 ]
  %22 = load i32, ptr %20, align 8, !tbaa !14
  %23 = load i32, ptr %21, align 8, !tbaa !14
  %24 = icmp eq i32 %22, %23
  br i1 %24, label %25, label %30

25:                                               ; preds = %19
  %26 = getelementptr inbounds nuw i8, ptr %21, i64 8
  %27 = load ptr, ptr %26, align 8, !tbaa !13
  %28 = icmp eq ptr %27, %1
  %29 = zext i1 %28 to i32
  br label %30

30:                                               ; preds = %6, %19, %25
  %31 = phi i32 [ %29, %25 ], [ 0, %19 ], [ 0, %6 ]
  ret i32 %31
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @list_print(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #3 {
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %4 = load ptr, ptr %3, align 8, !tbaa !13
  %5 = tail call i32 @puts(ptr noundef nonnull dereferenceable(1) %0)
  %6 = load i32, ptr %1, align 8, !tbaa !14
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %6)
  %8 = load ptr, ptr %3, align 8, !tbaa !13
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 8
  %10 = load ptr, ptr %9, align 8, !tbaa !13
  %11 = icmp eq ptr %10, %4
  br i1 %11, label %28, label %12

12:                                               ; preds = %2, %12
  %13 = phi ptr [ %26, %12 ], [ %10, %2 ]
  %14 = phi ptr [ %25, %12 ], [ %9, %2 ]
  %15 = phi ptr [ %24, %12 ], [ %8, %2 ]
  %16 = phi i32 [ %17, %12 ], [ 0, %2 ]
  %17 = add nuw nsw i32 %16, 1
  %18 = load i32, ptr %15, align 8, !tbaa !14
  %19 = load i32, ptr %13, align 8, !tbaa !14
  %20 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %21 = load ptr, ptr %20, align 8, !tbaa !6
  %22 = load i32, ptr %21, align 8, !tbaa !14
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %17, i32 noundef %18, i32 noundef %19, i32 noundef %22)
  %24 = load ptr, ptr %14, align 8, !tbaa !13
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 8
  %26 = load ptr, ptr %25, align 8, !tbaa !13
  %27 = icmp eq ptr %26, %4
  br i1 %27, label %28, label %12, !llvm.loop !17

28:                                               ; preds = %12, %2
  %29 = phi ptr [ %9, %2 ], [ %25, %12 ]
  %30 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %31 = load ptr, ptr %29, align 8, !tbaa !13
  %32 = load i32, ptr %31, align 8, !tbaa !14
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %32)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

; Function Attrs: mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noundef ptr @list_new() local_unnamed_addr #5 {
  %1 = tail call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #15
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %1, ptr %2, align 8, !tbaa !13
  %3 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store ptr %1, ptr %3, align 8, !tbaa !6
  store i32 0, ptr %1, align 8, !tbaa !14
  ret ptr %1
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #6

; Function Attrs: nofree nounwind memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noundef ptr @list_sequence(i32 noundef %0, i32 noundef %1) local_unnamed_addr #7 {
  %3 = tail call i32 @llvm.smax.i32(i32 %0, i32 %1)
  %4 = tail call i32 @llvm.smin.i32(i32 %0, i32 %1)
  %5 = sub nsw i32 %3, %4
  %6 = add nsw i32 %5, 2
  %7 = sext i32 %6 to i64
  %8 = mul nsw i64 %7, 24
  %9 = tail call noalias ptr @malloc(i64 noundef %8) #15
  %10 = add nsw i32 %4, -1
  %11 = icmp slt i32 %5, 0
  br i1 %11, label %66, label %12

12:                                               ; preds = %2
  %13 = add i32 %3, 1
  %14 = sub i32 %13, %4
  %15 = zext i32 %14 to i64
  %16 = icmp ult i32 %14, 2
  br i1 %16, label %48, label %17

17:                                               ; preds = %12
  %18 = and i64 %15, 4294967294
  %19 = or i64 %15, 1
  %20 = trunc nuw i64 %18 to i32
  %21 = add i32 %10, %20
  br label %22

22:                                               ; preds = %22, %17
  %23 = phi i64 [ 0, %17 ], [ %44, %22 ]
  %24 = or disjoint i64 %23, 1
  %25 = add i64 %23, 2
  %26 = trunc i64 %23 to i32
  %27 = add i32 %10, %26
  %28 = add i32 %4, %26
  %29 = getelementptr inbounds nuw %struct.DLL, ptr %9, i64 %23
  %30 = getelementptr inbounds nuw i8, ptr %29, i64 24
  %31 = getelementptr inbounds nuw %struct.DLL, ptr %9, i64 %23
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 48
  %33 = getelementptr inbounds nuw %struct.DLL, ptr %9, i64 %23
  %34 = getelementptr inbounds nuw %struct.DLL, ptr %9, i64 %23
  %35 = getelementptr inbounds nuw i8, ptr %34, i64 24
  %36 = getelementptr inbounds nuw i8, ptr %33, i64 8
  %37 = getelementptr inbounds nuw i8, ptr %34, i64 32
  store ptr %30, ptr %36, align 8, !tbaa !13
  store ptr %32, ptr %37, align 8, !tbaa !13
  %38 = getelementptr %struct.DLL, ptr %9, i64 %24
  %39 = getelementptr %struct.DLL, ptr %9, i64 %25
  %40 = getelementptr i8, ptr %38, i64 -24
  %41 = getelementptr i8, ptr %39, i64 -24
  %42 = getelementptr inbounds nuw %struct.DLL, ptr %9, i64 %24, i32 2
  %43 = getelementptr inbounds nuw %struct.DLL, ptr %9, i64 %25, i32 2
  store ptr %40, ptr %42, align 8, !tbaa !6
  store ptr %41, ptr %43, align 8, !tbaa !6
  store i32 %27, ptr %33, align 8, !tbaa !14
  store i32 %28, ptr %35, align 8, !tbaa !14
  %44 = add nuw i64 %23, 2
  %45 = icmp eq i64 %44, %18
  br i1 %45, label %46, label %22, !llvm.loop !18

46:                                               ; preds = %22
  %47 = icmp eq i64 %18, %15
  br i1 %47, label %66, label %48

48:                                               ; preds = %12, %46
  %49 = phi i64 [ 0, %12 ], [ %18, %46 ]
  %50 = phi i64 [ 1, %12 ], [ %19, %46 ]
  %51 = phi i32 [ %10, %12 ], [ %21, %46 ]
  br label %52

52:                                               ; preds = %48, %52
  %53 = phi i64 [ %56, %52 ], [ %49, %48 ]
  %54 = phi i64 [ %64, %52 ], [ %50, %48 ]
  %55 = phi i32 [ %63, %52 ], [ %51, %48 ]
  %56 = add nuw nsw i64 %53, 1
  %57 = getelementptr inbounds nuw %struct.DLL, ptr %9, i64 %56
  %58 = getelementptr inbounds nuw %struct.DLL, ptr %9, i64 %53
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 8
  store ptr %57, ptr %59, align 8, !tbaa !13
  %60 = getelementptr %struct.DLL, ptr %9, i64 %54
  %61 = getelementptr i8, ptr %60, i64 -24
  %62 = getelementptr inbounds nuw %struct.DLL, ptr %9, i64 %54, i32 2
  store ptr %61, ptr %62, align 8, !tbaa !6
  %63 = add nsw i32 %55, 1
  store i32 %55, ptr %58, align 8, !tbaa !14
  %64 = add nuw nsw i64 %54, 1
  %65 = icmp eq i64 %56, %15
  br i1 %65, label %66, label %52, !llvm.loop !21

66:                                               ; preds = %52, %46, %2
  %67 = phi i32 [ %10, %2 ], [ %21, %46 ], [ %63, %52 ]
  %68 = add nsw i32 %5, 1
  %69 = sext i32 %68 to i64
  %70 = getelementptr inbounds %struct.DLL, ptr %9, i64 %69
  %71 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store ptr %70, ptr %71, align 8, !tbaa !6
  %72 = getelementptr inbounds nuw i8, ptr %70, i64 8
  store ptr %9, ptr %72, align 8, !tbaa !13
  %73 = sext i32 %5 to i64
  %74 = getelementptr inbounds %struct.DLL, ptr %9, i64 %73
  %75 = getelementptr inbounds nuw i8, ptr %70, i64 16
  store ptr %74, ptr %75, align 8, !tbaa !6
  store i32 %67, ptr %70, align 8, !tbaa !14
  store i32 %68, ptr %9, align 8, !tbaa !14
  ret ptr %9
}

; Function Attrs: nofree nounwind memory(readwrite, argmem: read) uwtable
define dso_local noundef ptr @list_copy(ptr noundef readonly captures(none) %0) local_unnamed_addr #8 {
  %2 = load i32, ptr %0, align 8, !tbaa !14
  %3 = add nsw i32 %2, 1
  %4 = sext i32 %3 to i64
  %5 = mul nsw i64 %4, 24
  %6 = tail call noalias ptr @malloc(i64 noundef %5) #15
  %7 = icmp sgt i32 %2, 0
  br i1 %7, label %8, label %24

8:                                                ; preds = %1
  %9 = zext nneg i32 %2 to i64
  br label %10

10:                                               ; preds = %8, %10
  %11 = phi i64 [ 1, %8 ], [ %20, %10 ]
  %12 = phi i64 [ 0, %8 ], [ %19, %10 ]
  %13 = phi ptr [ %0, %8 ], [ %22, %10 ]
  %14 = getelementptr inbounds nuw %struct.DLL, ptr %6, i64 %11
  %15 = getelementptr inbounds nuw %struct.DLL, ptr %6, i64 %12
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store ptr %14, ptr %16, align 8, !tbaa !13
  %17 = getelementptr inbounds nuw i8, ptr %14, i64 16
  store ptr %15, ptr %17, align 8, !tbaa !6
  %18 = load i32, ptr %13, align 8, !tbaa !14
  store i32 %18, ptr %15, align 8, !tbaa !14
  %19 = add nuw nsw i64 %12, 1
  %20 = add nuw nsw i64 %11, 1
  %21 = getelementptr inbounds nuw i8, ptr %13, i64 8
  %22 = load ptr, ptr %21, align 8, !tbaa !13
  %23 = icmp eq i64 %19, %9
  br i1 %23, label %24, label %10, !llvm.loop !22

24:                                               ; preds = %10, %1
  %25 = sext i32 %2 to i64
  %26 = getelementptr inbounds %struct.DLL, ptr %6, i64 %25
  %27 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store ptr %26, ptr %27, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %26, i64 8
  store ptr %6, ptr %28, align 8, !tbaa !13
  %29 = getelementptr i8, ptr %0, i64 16
  %30 = load ptr, ptr %29, align 8, !tbaa !6
  %31 = load i32, ptr %30, align 8, !tbaa !14
  store i32 %31, ptr %26, align 8, !tbaa !14
  ret ptr %6
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @list_reverse(ptr noundef captures(address) %0) local_unnamed_addr #9 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi ptr [ %0, %1 ], [ %7, %2 ]
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %5 = load <2 x ptr>, ptr %4, align 8, !tbaa !23
  %6 = shufflevector <2 x ptr> %5, <2 x ptr> poison, <2 x i32> <i32 1, i32 0>
  store <2 x ptr> %6, ptr %4, align 8, !tbaa !23
  %7 = extractelement <2 x ptr> %5, i64 0
  %8 = icmp eq ptr %7, %0
  br i1 %8, label %9, label %2, !llvm.loop !24

9:                                                ; preds = %2
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @test_lists() local_unnamed_addr #10 {
  %1 = tail call noalias dereferenceable_or_null(2424) ptr @malloc(i64 noundef 2424) #15
  br label %2

2:                                                ; preds = %2, %0
  %3 = phi i64 [ 0, %0 ], [ %23, %2 ]
  %4 = or disjoint i64 %3, 1
  %5 = add i64 %3, 2
  %6 = trunc i64 %3 to i32
  %7 = or disjoint i32 %6, 1
  %8 = getelementptr inbounds nuw %struct.DLL, ptr %1, i64 %3
  %9 = getelementptr inbounds nuw i8, ptr %8, i64 24
  %10 = getelementptr inbounds nuw %struct.DLL, ptr %1, i64 %3
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 48
  %12 = getelementptr inbounds nuw %struct.DLL, ptr %1, i64 %3
  %13 = getelementptr inbounds nuw %struct.DLL, ptr %1, i64 %3
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 24
  %15 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %16 = getelementptr inbounds nuw i8, ptr %13, i64 32
  store ptr %9, ptr %15, align 8, !tbaa !13
  store ptr %11, ptr %16, align 8, !tbaa !13
  %17 = getelementptr %struct.DLL, ptr %1, i64 %4
  %18 = getelementptr %struct.DLL, ptr %1, i64 %5
  %19 = getelementptr i8, ptr %17, i64 -24
  %20 = getelementptr i8, ptr %18, i64 -24
  %21 = getelementptr inbounds nuw %struct.DLL, ptr %1, i64 %4, i32 2
  %22 = getelementptr inbounds nuw %struct.DLL, ptr %1, i64 %5, i32 2
  store ptr %19, ptr %21, align 8, !tbaa !6
  store ptr %20, ptr %22, align 8, !tbaa !6
  store i32 %6, ptr %12, align 8, !tbaa !14
  store i32 %7, ptr %14, align 8, !tbaa !14
  %23 = add nuw i64 %3, 2
  %24 = icmp eq i64 %23, 100
  br i1 %24, label %25, label %2, !llvm.loop !25

25:                                               ; preds = %2
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 2400
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store ptr %26, ptr %27, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 2408
  store ptr %1, ptr %28, align 8, !tbaa !13
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 2376
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 2416
  store ptr %29, ptr %30, align 8, !tbaa !6
  store i32 100, ptr %26, align 8, !tbaa !14
  store i32 100, ptr %1, align 8, !tbaa !14
  %31 = tail call noalias dereferenceable_or_null(2424) ptr @malloc(i64 noundef 2424) #15
  br label %32

32:                                               ; preds = %32, %25
  %33 = phi i64 [ 1, %25 ], [ %42, %32 ]
  %34 = phi i64 [ 0, %25 ], [ %41, %32 ]
  %35 = phi ptr [ %1, %25 ], [ %44, %32 ]
  %36 = getelementptr inbounds nuw %struct.DLL, ptr %31, i64 %33
  %37 = getelementptr inbounds nuw %struct.DLL, ptr %31, i64 %34
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 8
  store ptr %36, ptr %38, align 8, !tbaa !13
  %39 = getelementptr inbounds nuw i8, ptr %36, i64 16
  store ptr %37, ptr %39, align 8, !tbaa !6
  %40 = load i32, ptr %35, align 8, !tbaa !14
  store i32 %40, ptr %37, align 8, !tbaa !14
  %41 = add nuw nsw i64 %34, 1
  %42 = add nuw nsw i64 %33, 1
  %43 = getelementptr inbounds nuw i8, ptr %35, i64 8
  %44 = load ptr, ptr %43, align 8, !tbaa !13
  %45 = icmp eq i64 %41, 100
  br i1 %45, label %46, label %32, !llvm.loop !22

46:                                               ; preds = %32
  %47 = getelementptr inbounds nuw i8, ptr %31, i64 2400
  %48 = getelementptr inbounds nuw i8, ptr %31, i64 16
  store ptr %47, ptr %48, align 8, !tbaa !6
  %49 = getelementptr inbounds nuw i8, ptr %31, i64 2408
  store ptr %31, ptr %49, align 8, !tbaa !13
  store i32 100, ptr %47, align 8, !tbaa !14
  %50 = tail call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #15
  %51 = getelementptr inbounds nuw i8, ptr %50, i64 8
  store ptr %50, ptr %51, align 8, !tbaa !13
  %52 = getelementptr inbounds nuw i8, ptr %50, i64 16
  store ptr %50, ptr %52, align 8, !tbaa !6
  store i32 0, ptr %50, align 8, !tbaa !14
  %53 = getelementptr inbounds nuw i8, ptr %31, i64 8
  %54 = load ptr, ptr %53, align 8, !tbaa !13
  %55 = icmp eq ptr %54, %31
  br i1 %55, label %71, label %56

56:                                               ; preds = %46, %63
  %57 = phi ptr [ %67, %63 ], [ %54, %46 ]
  %58 = phi ptr [ %65, %63 ], [ %1, %46 ]
  %59 = phi ptr [ %57, %63 ], [ %31, %46 ]
  %60 = load i32, ptr %59, align 8, !tbaa !14
  %61 = load i32, ptr %58, align 8, !tbaa !14
  %62 = icmp eq i32 %60, %61
  br i1 %62, label %63, label %84

63:                                               ; preds = %56
  %64 = getelementptr inbounds nuw i8, ptr %58, i64 8
  %65 = load ptr, ptr %64, align 8, !tbaa !13
  %66 = getelementptr inbounds nuw i8, ptr %57, i64 8
  %67 = load ptr, ptr %66, align 8, !tbaa !13
  %68 = icmp eq ptr %67, %31
  br i1 %68, label %69, label %56, !llvm.loop !15

69:                                               ; preds = %63
  %70 = load i32, ptr %65, align 8, !tbaa !14
  br label %71

71:                                               ; preds = %69, %46
  %72 = phi i32 [ 100, %46 ], [ %70, %69 ]
  %73 = phi ptr [ %31, %46 ], [ %57, %69 ]
  %74 = phi ptr [ %1, %46 ], [ %65, %69 ]
  %75 = load i32, ptr %73, align 8, !tbaa !14
  %76 = icmp eq i32 %75, %72
  br i1 %76, label %77, label %84

77:                                               ; preds = %71
  %78 = getelementptr inbounds nuw i8, ptr %74, i64 8
  %79 = load ptr, ptr %78, align 8, !tbaa !13
  %80 = icmp eq ptr %79, %1
  br i1 %80, label %81, label %84

81:                                               ; preds = %77
  %82 = load i32, ptr %31, align 8, !tbaa !14
  %83 = icmp eq i32 %82, 0
  br i1 %83, label %98, label %86

84:                                               ; preds = %56, %71, %77
  %85 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.13)
  tail call void @exit(i32 noundef 1) #16
  unreachable

86:                                               ; preds = %81, %86
  %87 = phi i32 [ %92, %86 ], [ %82, %81 ]
  %88 = load ptr, ptr %53, align 8, !tbaa !13
  %89 = getelementptr inbounds nuw i8, ptr %88, i64 8
  %90 = load ptr, ptr %89, align 8, !tbaa !13
  store ptr %90, ptr %53, align 8, !tbaa !13
  %91 = getelementptr inbounds nuw i8, ptr %90, i64 16
  store ptr %31, ptr %91, align 8, !tbaa !6
  %92 = add nsw i32 %87, -1
  %93 = load ptr, ptr %52, align 8, !tbaa !6
  %94 = getelementptr inbounds nuw i8, ptr %93, i64 8
  store ptr %88, ptr %94, align 8, !tbaa !13
  store ptr %50, ptr %89, align 8, !tbaa !13
  store ptr %88, ptr %52, align 8, !tbaa !6
  %95 = getelementptr inbounds nuw i8, ptr %88, i64 16
  store ptr %93, ptr %95, align 8, !tbaa !6
  %96 = icmp eq i32 %92, 0
  br i1 %96, label %99, label %86, !llvm.loop !26

97:                                               ; preds = %99
  store i32 0, ptr %50, align 8, !tbaa !14
  store i32 %82, ptr %31, align 8, !tbaa !14
  br label %98

98:                                               ; preds = %81, %97
  br label %110

99:                                               ; preds = %86, %99
  %100 = phi i32 [ %105, %99 ], [ %82, %86 ]
  %101 = load ptr, ptr %52, align 8, !tbaa !6
  %102 = getelementptr inbounds nuw i8, ptr %101, i64 16
  %103 = load ptr, ptr %102, align 8, !tbaa !6
  %104 = getelementptr inbounds nuw i8, ptr %103, i64 8
  store ptr %50, ptr %104, align 8, !tbaa !13
  store ptr %103, ptr %52, align 8, !tbaa !6
  %105 = add nsw i32 %100, -1
  %106 = load ptr, ptr %48, align 8, !tbaa !6
  %107 = getelementptr inbounds nuw i8, ptr %106, i64 8
  store ptr %101, ptr %107, align 8, !tbaa !13
  %108 = getelementptr inbounds nuw i8, ptr %101, i64 8
  store ptr %31, ptr %108, align 8, !tbaa !13
  store ptr %101, ptr %48, align 8, !tbaa !6
  store ptr %106, ptr %102, align 8, !tbaa !6
  %109 = icmp eq i32 %105, 0
  br i1 %109, label %97, label %99, !llvm.loop !27

110:                                              ; preds = %98, %110
  %111 = phi ptr [ %115, %110 ], [ %1, %98 ]
  %112 = getelementptr inbounds nuw i8, ptr %111, i64 8
  %113 = load <2 x ptr>, ptr %112, align 8, !tbaa !23
  %114 = shufflevector <2 x ptr> %113, <2 x ptr> poison, <2 x i32> <i32 1, i32 0>
  store <2 x ptr> %114, ptr %112, align 8, !tbaa !23
  %115 = extractelement <2 x ptr> %113, i64 0
  %116 = icmp eq ptr %115, %1
  br i1 %116, label %117, label %110, !llvm.loop !24

117:                                              ; preds = %110
  %118 = getelementptr i8, ptr %1, i64 8
  %119 = load ptr, ptr %118, align 8, !tbaa !13
  %120 = load i32, ptr %119, align 8, !tbaa !14
  %121 = icmp eq i32 %120, 100
  br i1 %121, label %124, label %122

122:                                              ; preds = %117
  %123 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef 100, i32 noundef %120)
  tail call void @exit(i32 noundef 1) #16
  unreachable

124:                                              ; preds = %117
  %125 = load ptr, ptr %27, align 8, !tbaa !6
  %126 = load i32, ptr %125, align 8, !tbaa !14
  %127 = icmp eq i32 %126, 1
  br i1 %127, label %130, label %128

128:                                              ; preds = %124
  %129 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef 100, i32 noundef %126)
  tail call void @exit(i32 noundef 1) #16
  unreachable

130:                                              ; preds = %124
  %131 = load ptr, ptr %53, align 8, !tbaa !13
  %132 = load i32, ptr %131, align 8, !tbaa !14
  %133 = icmp eq i32 %132, 100
  br i1 %133, label %136, label %134

134:                                              ; preds = %130
  %135 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef 100, i32 noundef %132)
  tail call void @exit(i32 noundef 1) #16
  unreachable

136:                                              ; preds = %130
  %137 = load ptr, ptr %48, align 8, !tbaa !6
  %138 = load i32, ptr %137, align 8, !tbaa !14
  %139 = icmp eq i32 %138, 1
  br i1 %139, label %142, label %140

140:                                              ; preds = %136
  %141 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef 100, i32 noundef %138)
  tail call void @exit(i32 noundef 1) #16
  unreachable

142:                                              ; preds = %136
  %143 = icmp eq ptr %119, %1
  br i1 %143, label %160, label %144

144:                                              ; preds = %142, %151
  %145 = phi ptr [ %155, %151 ], [ %119, %142 ]
  %146 = phi ptr [ %153, %151 ], [ %31, %142 ]
  %147 = phi ptr [ %145, %151 ], [ %1, %142 ]
  %148 = load i32, ptr %147, align 8, !tbaa !14
  %149 = load i32, ptr %146, align 8, !tbaa !14
  %150 = icmp eq i32 %148, %149
  br i1 %150, label %151, label %169

151:                                              ; preds = %144
  %152 = getelementptr inbounds nuw i8, ptr %146, i64 8
  %153 = load ptr, ptr %152, align 8, !tbaa !13
  %154 = getelementptr inbounds nuw i8, ptr %145, i64 8
  %155 = load ptr, ptr %154, align 8, !tbaa !13
  %156 = icmp eq ptr %155, %1
  br i1 %156, label %157, label %144, !llvm.loop !15

157:                                              ; preds = %151
  %158 = load i32, ptr %145, align 8, !tbaa !14
  %159 = load i32, ptr %153, align 8, !tbaa !14
  br label %160

160:                                              ; preds = %157, %142
  %161 = phi i32 [ %82, %142 ], [ %159, %157 ]
  %162 = phi i32 [ 100, %142 ], [ %158, %157 ]
  %163 = phi ptr [ %31, %142 ], [ %153, %157 ]
  %164 = icmp eq i32 %162, %161
  br i1 %164, label %165, label %169

165:                                              ; preds = %160
  %166 = getelementptr inbounds nuw i8, ptr %163, i64 8
  %167 = load ptr, ptr %166, align 8, !tbaa !13
  %168 = icmp eq ptr %167, %31
  br i1 %168, label %171, label %169

169:                                              ; preds = %144, %160, %165
  %170 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.16)
  tail call void @exit(i32 noundef 1) #16
  unreachable

171:                                              ; preds = %165
  tail call void @free(ptr noundef %1) #17
  tail call void @free(ptr noundef %31) #17
  tail call void @free(ptr noundef %50) #17
  ret i32 100
}

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #11

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #12

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #10 {
  %3 = icmp eq i32 %0, 2
  br i1 %3, label %4, label %10

4:                                                ; preds = %2
  %5 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !28
  %7 = tail call i64 @strtol(ptr noundef nonnull captures(none) %6, ptr noundef null, i32 noundef 10) #17
  %8 = trunc i64 %7 to i32
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %17, label %10

10:                                               ; preds = %2, %4
  %11 = phi i32 [ 3000000, %2 ], [ %8, %4 ]
  br label %12

12:                                               ; preds = %10, %12
  %13 = phi i32 [ %14, %12 ], [ %11, %10 ]
  %14 = add nsw i32 %13, -1
  %15 = tail call i32 @test_lists()
  %16 = icmp eq i32 %14, 0
  br i1 %16, label %17, label %12, !llvm.loop !30

17:                                               ; preds = %12, %4
  %18 = phi i32 [ 0, %4 ], [ %15, %12 ]
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i32 noundef %18)
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #13

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #14

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #14

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nofree nounwind willreturn memory(write, argmem: none, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree nounwind memory(write, argmem: none, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree nounwind memory(readwrite, argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #15 = { nounwind allocsize(0) }
attributes #16 = { cold noreturn nounwind }
attributes #17 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !11, i64 16}
!7 = !{!"DLL", !8, i64 0, !11, i64 8, !11, i64 16}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!"p1 _ZTS3DLL", !12, i64 0}
!12 = !{!"any pointer", !9, i64 0}
!13 = !{!7, !11, i64 8}
!14 = !{!7, !8, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
!17 = distinct !{!17, !16}
!18 = distinct !{!18, !16, !19, !20}
!19 = !{!"llvm.loop.isvectorized", i32 1}
!20 = !{!"llvm.loop.unroll.runtime.disable"}
!21 = distinct !{!21, !16, !19}
!22 = distinct !{!22, !16}
!23 = !{!11, !11, i64 0}
!24 = distinct !{!24, !16}
!25 = distinct !{!25, !16, !19, !20}
!26 = distinct !{!26, !16}
!27 = distinct !{!27, !16}
!28 = !{!29, !29, i64 0}
!29 = !{!"p1 omnipotent char", !12, i64 0}
!30 = distinct !{!30, !16}
