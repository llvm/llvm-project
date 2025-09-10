; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/gcc-loops.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Vectorizer/gcc-loops.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

module asm ".globl _ZSt21ios_base_library_initv"

%struct.A = type { [1024 x i32] }
%"class.std::basic_ostream" = type { ptr, %"class.std::basic_ios" }
%"class.std::basic_ios" = type { %"class.std::ios_base", ptr, i8, i8, ptr, ptr, ptr, ptr }
%"class.std::ios_base" = type { ptr, i64, i64, i32, i32, i32, ptr, %"struct.std::ios_base::_Words", [8 x %"struct.std::ios_base::_Words"], i32, ptr, %"class.std::locale" }
%"struct.std::ios_base::_Words" = type { ptr, i64 }
%"class.std::locale" = type { ptr }
%class.Timer = type { ptr, i8, %struct.timeval, %struct.timeval }
%struct.timeval = type { i64, i64 }

$_ZN5TimerD2Ev = comdat any

$__clang_call_terminate = comdat any

@usa = dso_local global [1024 x i16] zeroinitializer, align 2
@sa = dso_local global [1024 x i16] zeroinitializer, align 2
@sb = dso_local global [1024 x i16] zeroinitializer, align 2
@sc = dso_local global [1024 x i16] zeroinitializer, align 2
@ua = dso_local global [1024 x i32] zeroinitializer, align 4
@ia = dso_local global [1024 x i32] zeroinitializer, align 16
@ib = dso_local global [1024 x i32] zeroinitializer, align 16
@ic = dso_local global [1024 x i32] zeroinitializer, align 16
@ub = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 4
@uc = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 4
@fa = dso_local local_unnamed_addr global [1024 x float] zeroinitializer, align 4
@fb = dso_local local_unnamed_addr global [1024 x float] zeroinitializer, align 4
@da = dso_local local_unnamed_addr global [1024 x float] zeroinitializer, align 4
@db = dso_local local_unnamed_addr global [1024 x float] zeroinitializer, align 4
@dc = dso_local local_unnamed_addr global [1024 x float] zeroinitializer, align 4
@dd = dso_local local_unnamed_addr global [1024 x float] zeroinitializer, align 4
@dj = dso_local local_unnamed_addr global [1024 x i32] zeroinitializer, align 4
@s = dso_local local_unnamed_addr global %struct.A zeroinitializer, align 4
@a = dso_local local_unnamed_addr global [2048 x i32] zeroinitializer, align 16
@b = dso_local local_unnamed_addr global [2048 x i32] zeroinitializer, align 16
@c = dso_local local_unnamed_addr global [2048 x i32] zeroinitializer, align 16
@d = dso_local local_unnamed_addr global [2048 x i32] zeroinitializer, align 16
@G = dso_local local_unnamed_addr global [32 x [1024 x i32]] zeroinitializer, align 4
@.str = private unnamed_addr constant [9 x i8] c"Example1\00", align 1
@.str.1 = private unnamed_addr constant [10 x i8] c"Example2a\00", align 1
@.str.2 = private unnamed_addr constant [10 x i8] c"Example2b\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"Example3\00", align 1
@.str.4 = private unnamed_addr constant [10 x i8] c"Example4a\00", align 1
@.str.5 = private unnamed_addr constant [10 x i8] c"Example4b\00", align 1
@.str.6 = private unnamed_addr constant [10 x i8] c"Example4c\00", align 1
@.str.7 = private unnamed_addr constant [9 x i8] c"Example7\00", align 1
@.str.8 = private unnamed_addr constant [9 x i8] c"Example8\00", align 1
@.str.9 = private unnamed_addr constant [9 x i8] c"Example9\00", align 1
@.str.10 = private unnamed_addr constant [11 x i8] c"Example10a\00", align 1
@.str.11 = private unnamed_addr constant [11 x i8] c"Example10b\00", align 1
@.str.12 = private unnamed_addr constant [10 x i8] c"Example11\00", align 1
@.str.13 = private unnamed_addr constant [10 x i8] c"Example12\00", align 1
@.str.14 = private unnamed_addr constant [10 x i8] c"Example23\00", align 1
@.str.15 = private unnamed_addr constant [10 x i8] c"Example24\00", align 1
@.str.16 = private unnamed_addr constant [10 x i8] c"Example25\00", align 1
@_ZSt4cout = external global %"class.std::basic_ostream", align 8
@.str.17 = private unnamed_addr constant [11 x i8] c"Results: (\00", align 1
@.str.18 = private unnamed_addr constant [3 x i8] c"):\00", align 1
@.str.19 = private unnamed_addr constant [2 x i8] c" \00", align 1
@.str.20 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.21 = private unnamed_addr constant [3 x i8] c", \00", align 1
@.str.22 = private unnamed_addr constant [8 x i8] c", msec\0A\00", align 1
@.str.23 = private unnamed_addr constant [26 x i8] c"vector::_M_realloc_append\00", align 1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z8example1v() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %15, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @b, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 16, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 16, !tbaa !6
  %7 = getelementptr inbounds nuw i32, ptr @c, i64 %2
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %9 = load <4 x i32>, ptr %7, align 16, !tbaa !6
  %10 = load <4 x i32>, ptr %8, align 16, !tbaa !6
  %11 = add nsw <4 x i32> %9, %5
  %12 = add nsw <4 x i32> %10, %6
  %13 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 16
  store <4 x i32> %11, ptr %13, align 16, !tbaa !6
  store <4 x i32> %12, ptr %14, align 16, !tbaa !6
  %15 = add nuw i64 %2, 8
  %16 = icmp eq i64 %15, 256
  br i1 %16, label %17, label %1, !llvm.loop !10

17:                                               ; preds = %1
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z9example2aii(i32 noundef %0, i32 noundef %1) local_unnamed_addr #2 {
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %26

4:                                                ; preds = %2
  %5 = zext nneg i32 %0 to i64
  %6 = icmp ult i32 %0, 8
  br i1 %6, label %19, label %7

7:                                                ; preds = %4
  %8 = and i64 %5, 2147483640
  %9 = insertelement <4 x i32> poison, i32 %1, i64 0
  %10 = shufflevector <4 x i32> %9, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %11

11:                                               ; preds = %11, %7
  %12 = phi i64 [ 0, %7 ], [ %15, %11 ]
  %13 = getelementptr inbounds nuw i32, ptr @b, i64 %12
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 16
  store <4 x i32> %10, ptr %13, align 16, !tbaa !6
  store <4 x i32> %10, ptr %14, align 16, !tbaa !6
  %15 = add nuw i64 %12, 8
  %16 = icmp eq i64 %15, %8
  br i1 %16, label %17, label %11, !llvm.loop !14

17:                                               ; preds = %11
  %18 = icmp eq i64 %8, %5
  br i1 %18, label %26, label %19

19:                                               ; preds = %4, %17
  %20 = phi i64 [ 0, %4 ], [ %8, %17 ]
  br label %21

21:                                               ; preds = %19, %21
  %22 = phi i64 [ %24, %21 ], [ %20, %19 ]
  %23 = getelementptr inbounds nuw i32, ptr @b, i64 %22
  store i32 %1, ptr %23, align 4, !tbaa !6
  %24 = add nuw nsw i64 %22, 1
  %25 = icmp eq i64 %24, %5
  br i1 %25, label %26, label %21, !llvm.loop !15

26:                                               ; preds = %21, %17, %2
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z9example2bii(i32 noundef %0, i32 %1) local_unnamed_addr #0 {
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %44, label %4

4:                                                ; preds = %2
  %5 = zext i32 %0 to i64
  %6 = icmp ult i32 %0, 8
  br i1 %6, label %29, label %7

7:                                                ; preds = %4
  %8 = and i64 %5, 4294967288
  %9 = trunc nuw i64 %8 to i32
  %10 = sub i32 %0, %9
  br label %11

11:                                               ; preds = %11, %7
  %12 = phi i64 [ 0, %7 ], [ %25, %11 ]
  %13 = getelementptr inbounds nuw i32, ptr @b, i64 %12
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %15 = load <4 x i32>, ptr %13, align 16, !tbaa !6
  %16 = load <4 x i32>, ptr %14, align 16, !tbaa !6
  %17 = getelementptr inbounds nuw i32, ptr @c, i64 %12
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %19 = load <4 x i32>, ptr %17, align 16, !tbaa !6
  %20 = load <4 x i32>, ptr %18, align 16, !tbaa !6
  %21 = and <4 x i32> %19, %15
  %22 = and <4 x i32> %20, %16
  %23 = getelementptr inbounds nuw i32, ptr @a, i64 %12
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  store <4 x i32> %21, ptr %23, align 16, !tbaa !6
  store <4 x i32> %22, ptr %24, align 16, !tbaa !6
  %25 = add nuw i64 %12, 8
  %26 = icmp eq i64 %25, %8
  br i1 %26, label %27, label %11, !llvm.loop !16

27:                                               ; preds = %11
  %28 = icmp eq i64 %8, %5
  br i1 %28, label %44, label %29

29:                                               ; preds = %4, %27
  %30 = phi i64 [ 0, %4 ], [ %8, %27 ]
  %31 = phi i32 [ %0, %4 ], [ %10, %27 ]
  br label %32

32:                                               ; preds = %29, %32
  %33 = phi i64 [ %42, %32 ], [ %30, %29 ]
  %34 = phi i32 [ %35, %32 ], [ %31, %29 ]
  %35 = add nsw i32 %34, -1
  %36 = getelementptr inbounds nuw i32, ptr @b, i64 %33
  %37 = load i32, ptr %36, align 4, !tbaa !6
  %38 = getelementptr inbounds nuw i32, ptr @c, i64 %33
  %39 = load i32, ptr %38, align 4, !tbaa !6
  %40 = and i32 %39, %37
  %41 = getelementptr inbounds nuw i32, ptr @a, i64 %33
  store i32 %40, ptr %41, align 4, !tbaa !6
  %42 = add nuw nsw i64 %33, 1
  %43 = icmp eq i32 %35, 0
  br i1 %43, label %44, label %32, !llvm.loop !17

44:                                               ; preds = %32, %27, %2
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @_Z8example3iPiS_(i32 noundef %0, ptr noalias noundef writeonly captures(none) %1, ptr noalias noundef readonly captures(none) %2) local_unnamed_addr #3 {
  %4 = icmp eq i32 %0, 0
  br i1 %4, label %8, label %5

5:                                                ; preds = %3
  %6 = zext i32 %0 to i64
  %7 = shl nuw nsw i64 %6, 2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 16 %1, ptr align 16 %2, i64 %7, i1 false), !tbaa !6
  br label %8

8:                                                ; preds = %5, %3
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z9example4aiPiS_(i32 noundef %0, ptr noalias noundef writeonly captures(none) %1, ptr noalias noundef readonly captures(none) %2) local_unnamed_addr #4 {
  %4 = icmp eq i32 %0, 0
  br i1 %4, label %46, label %5

5:                                                ; preds = %3
  %6 = zext i32 %0 to i64
  %7 = icmp ult i32 %0, 8
  br i1 %7, label %32, label %8

8:                                                ; preds = %5
  %9 = and i64 %6, 4294967288
  %10 = shl nuw nsw i64 %9, 2
  %11 = getelementptr i8, ptr %2, i64 %10
  %12 = shl nuw nsw i64 %9, 2
  %13 = getelementptr i8, ptr %1, i64 %12
  %14 = trunc nuw i64 %9 to i32
  %15 = sub i32 %0, %14
  br label %16

16:                                               ; preds = %16, %8
  %17 = phi i64 [ 0, %8 ], [ %28, %16 ]
  %18 = shl i64 %17, 2
  %19 = getelementptr i8, ptr %2, i64 %18
  %20 = shl i64 %17, 2
  %21 = getelementptr i8, ptr %1, i64 %20
  %22 = getelementptr i8, ptr %19, i64 16
  %23 = load <4 x i32>, ptr %19, align 16, !tbaa !6
  %24 = load <4 x i32>, ptr %22, align 16, !tbaa !6
  %25 = add nsw <4 x i32> %23, splat (i32 5)
  %26 = add nsw <4 x i32> %24, splat (i32 5)
  %27 = getelementptr i8, ptr %21, i64 16
  store <4 x i32> %25, ptr %21, align 16, !tbaa !6
  store <4 x i32> %26, ptr %27, align 16, !tbaa !6
  %28 = add nuw i64 %17, 8
  %29 = icmp eq i64 %28, %9
  br i1 %29, label %30, label %16, !llvm.loop !18

30:                                               ; preds = %16
  %31 = icmp eq i64 %9, %6
  br i1 %31, label %46, label %32

32:                                               ; preds = %5, %30
  %33 = phi ptr [ %2, %5 ], [ %11, %30 ]
  %34 = phi ptr [ %1, %5 ], [ %13, %30 ]
  %35 = phi i32 [ %0, %5 ], [ %15, %30 ]
  br label %36

36:                                               ; preds = %32, %36
  %37 = phi ptr [ %41, %36 ], [ %33, %32 ]
  %38 = phi ptr [ %44, %36 ], [ %34, %32 ]
  %39 = phi i32 [ %40, %36 ], [ %35, %32 ]
  %40 = add nsw i32 %39, -1
  %41 = getelementptr inbounds nuw i8, ptr %37, i64 4
  %42 = load i32, ptr %37, align 16, !tbaa !6
  %43 = add nsw i32 %42, 5
  %44 = getelementptr inbounds nuw i8, ptr %38, i64 4
  store i32 %43, ptr %38, align 16, !tbaa !6
  %45 = icmp eq i32 %40, 0
  br i1 %45, label %46, label %36, !llvm.loop !19

46:                                               ; preds = %36, %30, %3
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z9example4biPiS_(i32 noundef %0, ptr noalias readnone captures(none) %1, ptr noalias readnone captures(none) %2) local_unnamed_addr #0 {
  %4 = icmp sgt i32 %0, 0
  br i1 %4, label %5, label %43

5:                                                ; preds = %3
  %6 = zext nneg i32 %0 to i64
  %7 = icmp ult i32 %0, 8
  br i1 %7, label %30, label %8

8:                                                ; preds = %5
  %9 = and i64 %6, 2147483640
  br label %10

10:                                               ; preds = %10, %8
  %11 = phi i64 [ 0, %8 ], [ %26, %10 ]
  %12 = getelementptr inbounds nuw i32, ptr @b, i64 %11
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 4
  %14 = getelementptr inbounds nuw i8, ptr %12, i64 20
  %15 = load <4 x i32>, ptr %13, align 4, !tbaa !6
  %16 = load <4 x i32>, ptr %14, align 4, !tbaa !6
  %17 = getelementptr inbounds nuw i32, ptr @c, i64 %11
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 12
  %19 = getelementptr inbounds nuw i8, ptr %17, i64 28
  %20 = load <4 x i32>, ptr %18, align 4, !tbaa !6
  %21 = load <4 x i32>, ptr %19, align 4, !tbaa !6
  %22 = add nsw <4 x i32> %20, %15
  %23 = add nsw <4 x i32> %21, %16
  %24 = getelementptr inbounds nuw i32, ptr @a, i64 %11
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  store <4 x i32> %22, ptr %24, align 16, !tbaa !6
  store <4 x i32> %23, ptr %25, align 16, !tbaa !6
  %26 = add nuw i64 %11, 8
  %27 = icmp eq i64 %26, %9
  br i1 %27, label %28, label %10, !llvm.loop !20

28:                                               ; preds = %10
  %29 = icmp eq i64 %9, %6
  br i1 %29, label %43, label %30

30:                                               ; preds = %5, %28
  %31 = phi i64 [ 0, %5 ], [ %9, %28 ]
  br label %32

32:                                               ; preds = %30, %32
  %33 = phi i64 [ %34, %32 ], [ %31, %30 ]
  %34 = add nuw nsw i64 %33, 1
  %35 = getelementptr inbounds nuw i32, ptr @b, i64 %34
  %36 = load i32, ptr %35, align 4, !tbaa !6
  %37 = getelementptr inbounds nuw i32, ptr @c, i64 %33
  %38 = getelementptr inbounds nuw i8, ptr %37, i64 12
  %39 = load i32, ptr %38, align 4, !tbaa !6
  %40 = add nsw i32 %39, %36
  %41 = getelementptr inbounds nuw i32, ptr @a, i64 %33
  store i32 %40, ptr %41, align 4, !tbaa !6
  %42 = icmp eq i64 %34, %6
  br i1 %42, label %43, label %32, !llvm.loop !21

43:                                               ; preds = %32, %28, %3
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z9example4ciPiS_(i32 noundef %0, ptr noalias readnone captures(none) %1, ptr noalias readnone captures(none) %2) local_unnamed_addr #0 {
  %4 = icmp sgt i32 %0, 0
  br i1 %4, label %5, label %37

5:                                                ; preds = %3
  %6 = zext nneg i32 %0 to i64
  %7 = icmp ult i32 %0, 8
  br i1 %7, label %26, label %8

8:                                                ; preds = %5
  %9 = and i64 %6, 2147483640
  br label %10

10:                                               ; preds = %10, %8
  %11 = phi i64 [ 0, %8 ], [ %22, %10 ]
  %12 = getelementptr inbounds nuw i32, ptr @a, i64 %11
  %13 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %14 = load <4 x i32>, ptr %12, align 16, !tbaa !6
  %15 = load <4 x i32>, ptr %13, align 16, !tbaa !6
  %16 = icmp sgt <4 x i32> %14, splat (i32 4)
  %17 = icmp sgt <4 x i32> %15, splat (i32 4)
  %18 = select <4 x i1> %16, <4 x i32> splat (i32 4), <4 x i32> zeroinitializer
  %19 = select <4 x i1> %17, <4 x i32> splat (i32 4), <4 x i32> zeroinitializer
  %20 = getelementptr inbounds nuw i32, ptr @b, i64 %11
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 16
  store <4 x i32> %18, ptr %20, align 16, !tbaa !6
  store <4 x i32> %19, ptr %21, align 16, !tbaa !6
  %22 = add nuw i64 %11, 8
  %23 = icmp eq i64 %22, %9
  br i1 %23, label %24, label %10, !llvm.loop !22

24:                                               ; preds = %10
  %25 = icmp eq i64 %9, %6
  br i1 %25, label %37, label %26

26:                                               ; preds = %5, %24
  %27 = phi i64 [ 0, %5 ], [ %9, %24 ]
  br label %28

28:                                               ; preds = %26, %28
  %29 = phi i64 [ %35, %28 ], [ %27, %26 ]
  %30 = getelementptr inbounds nuw i32, ptr @a, i64 %29
  %31 = load i32, ptr %30, align 4, !tbaa !6
  %32 = icmp sgt i32 %31, 4
  %33 = select i1 %32, i32 4, i32 0
  %34 = getelementptr inbounds nuw i32, ptr @b, i64 %29
  store i32 %33, ptr %34, align 4, !tbaa !6
  %35 = add nuw nsw i64 %29, 1
  %36 = icmp eq i64 %35, %6
  br i1 %36, label %37, label %28, !llvm.loop !23

37:                                               ; preds = %28, %24, %3
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: write) uwtable
define dso_local void @_Z8example5iP1A(i32 noundef %0, ptr noundef writeonly captures(none) %1) local_unnamed_addr #5 {
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %24

4:                                                ; preds = %2
  %5 = zext nneg i32 %0 to i64
  %6 = icmp ult i32 %0, 8
  br i1 %6, label %17, label %7

7:                                                ; preds = %4
  %8 = and i64 %5, 2147483640
  br label %9

9:                                                ; preds = %9, %7
  %10 = phi i64 [ 0, %7 ], [ %13, %9 ]
  %11 = getelementptr inbounds nuw i32, ptr %1, i64 %10
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 16
  store <4 x i32> splat (i32 5), ptr %11, align 4, !tbaa !6
  store <4 x i32> splat (i32 5), ptr %12, align 4, !tbaa !6
  %13 = add nuw i64 %10, 8
  %14 = icmp eq i64 %13, %8
  br i1 %14, label %15, label %9, !llvm.loop !24

15:                                               ; preds = %9
  %16 = icmp eq i64 %8, %5
  br i1 %16, label %24, label %17

17:                                               ; preds = %4, %15
  %18 = phi i64 [ 0, %4 ], [ %8, %15 ]
  br label %19

19:                                               ; preds = %17, %19
  %20 = phi i64 [ %22, %19 ], [ %18, %17 ]
  %21 = getelementptr inbounds nuw i32, ptr %1, i64 %20
  store i32 5, ptr %21, align 4, !tbaa !6
  %22 = add nuw nsw i64 %20, 1
  %23 = icmp eq i64 %22, %5
  br i1 %23, label %24, label %19, !llvm.loop !25

24:                                               ; preds = %19, %15, %2
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z8example7i(i32 noundef %0) local_unnamed_addr #6 {
  %2 = sext i32 %0 to i64
  %3 = shl nsw i64 %2, 2
  %4 = getelementptr i8, ptr @b, i64 %3
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 16 dereferenceable(4096) @a, ptr noundef nonnull align 4 dereferenceable(4096) %4, i64 4096, i1 false), !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z8example8i(i32 noundef %0) local_unnamed_addr #2 {
  %2 = insertelement <4 x i32> poison, i32 %0, i64 0
  %3 = shufflevector <4 x i32> %2, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %4

4:                                                ; preds = %4, %1
  %5 = phi i64 [ 0, %1 ], [ %8, %4 ]
  %6 = getelementptr inbounds nuw i32, ptr @G, i64 %5
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store <4 x i32> %3, ptr %6, align 4, !tbaa !6
  store <4 x i32> %3, ptr %7, align 4, !tbaa !6
  %8 = add nuw i64 %5, 8
  %9 = icmp eq i64 %8, 1024
  br i1 %9, label %10, label %4, !llvm.loop !26

10:                                               ; preds = %4
  %11 = insertelement <4 x i32> poison, i32 %0, i64 0
  %12 = shufflevector <4 x i32> %11, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi i64 [ 0, %10 ], [ %17, %13 ]
  %15 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 4096), i64 %14
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 16
  store <4 x i32> %12, ptr %15, align 4, !tbaa !6
  store <4 x i32> %12, ptr %16, align 4, !tbaa !6
  %17 = add nuw i64 %14, 8
  %18 = icmp eq i64 %17, 1024
  br i1 %18, label %19, label %13, !llvm.loop !27

19:                                               ; preds = %13
  %20 = insertelement <4 x i32> poison, i32 %0, i64 0
  %21 = shufflevector <4 x i32> %20, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %22

22:                                               ; preds = %22, %19
  %23 = phi i64 [ 0, %19 ], [ %26, %22 ]
  %24 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 8192), i64 %23
  %25 = getelementptr inbounds nuw i8, ptr %24, i64 16
  store <4 x i32> %21, ptr %24, align 4, !tbaa !6
  store <4 x i32> %21, ptr %25, align 4, !tbaa !6
  %26 = add nuw i64 %23, 8
  %27 = icmp eq i64 %26, 1024
  br i1 %27, label %28, label %22, !llvm.loop !28

28:                                               ; preds = %22
  %29 = insertelement <4 x i32> poison, i32 %0, i64 0
  %30 = shufflevector <4 x i32> %29, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %31

31:                                               ; preds = %31, %28
  %32 = phi i64 [ 0, %28 ], [ %35, %31 ]
  %33 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 12288), i64 %32
  %34 = getelementptr inbounds nuw i8, ptr %33, i64 16
  store <4 x i32> %30, ptr %33, align 4, !tbaa !6
  store <4 x i32> %30, ptr %34, align 4, !tbaa !6
  %35 = add nuw i64 %32, 8
  %36 = icmp eq i64 %35, 1024
  br i1 %36, label %37, label %31, !llvm.loop !29

37:                                               ; preds = %31
  %38 = insertelement <4 x i32> poison, i32 %0, i64 0
  %39 = shufflevector <4 x i32> %38, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %40

40:                                               ; preds = %40, %37
  %41 = phi i64 [ 0, %37 ], [ %44, %40 ]
  %42 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 16384), i64 %41
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 16
  store <4 x i32> %39, ptr %42, align 4, !tbaa !6
  store <4 x i32> %39, ptr %43, align 4, !tbaa !6
  %44 = add nuw i64 %41, 8
  %45 = icmp eq i64 %44, 1024
  br i1 %45, label %46, label %40, !llvm.loop !30

46:                                               ; preds = %40
  %47 = insertelement <4 x i32> poison, i32 %0, i64 0
  %48 = shufflevector <4 x i32> %47, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %49

49:                                               ; preds = %49, %46
  %50 = phi i64 [ 0, %46 ], [ %53, %49 ]
  %51 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 20480), i64 %50
  %52 = getelementptr inbounds nuw i8, ptr %51, i64 16
  store <4 x i32> %48, ptr %51, align 4, !tbaa !6
  store <4 x i32> %48, ptr %52, align 4, !tbaa !6
  %53 = add nuw i64 %50, 8
  %54 = icmp eq i64 %53, 1024
  br i1 %54, label %55, label %49, !llvm.loop !31

55:                                               ; preds = %49
  %56 = insertelement <4 x i32> poison, i32 %0, i64 0
  %57 = shufflevector <4 x i32> %56, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %58

58:                                               ; preds = %58, %55
  %59 = phi i64 [ 0, %55 ], [ %62, %58 ]
  %60 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 24576), i64 %59
  %61 = getelementptr inbounds nuw i8, ptr %60, i64 16
  store <4 x i32> %57, ptr %60, align 4, !tbaa !6
  store <4 x i32> %57, ptr %61, align 4, !tbaa !6
  %62 = add nuw i64 %59, 8
  %63 = icmp eq i64 %62, 1024
  br i1 %63, label %64, label %58, !llvm.loop !32

64:                                               ; preds = %58
  %65 = insertelement <4 x i32> poison, i32 %0, i64 0
  %66 = shufflevector <4 x i32> %65, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %67

67:                                               ; preds = %67, %64
  %68 = phi i64 [ 0, %64 ], [ %71, %67 ]
  %69 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 28672), i64 %68
  %70 = getelementptr inbounds nuw i8, ptr %69, i64 16
  store <4 x i32> %66, ptr %69, align 4, !tbaa !6
  store <4 x i32> %66, ptr %70, align 4, !tbaa !6
  %71 = add nuw i64 %68, 8
  %72 = icmp eq i64 %71, 1024
  br i1 %72, label %73, label %67, !llvm.loop !33

73:                                               ; preds = %67
  %74 = insertelement <4 x i32> poison, i32 %0, i64 0
  %75 = shufflevector <4 x i32> %74, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %76

76:                                               ; preds = %76, %73
  %77 = phi i64 [ 0, %73 ], [ %80, %76 ]
  %78 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 32768), i64 %77
  %79 = getelementptr inbounds nuw i8, ptr %78, i64 16
  store <4 x i32> %75, ptr %78, align 4, !tbaa !6
  store <4 x i32> %75, ptr %79, align 4, !tbaa !6
  %80 = add nuw i64 %77, 8
  %81 = icmp eq i64 %80, 1024
  br i1 %81, label %82, label %76, !llvm.loop !34

82:                                               ; preds = %76
  %83 = insertelement <4 x i32> poison, i32 %0, i64 0
  %84 = shufflevector <4 x i32> %83, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %85

85:                                               ; preds = %85, %82
  %86 = phi i64 [ 0, %82 ], [ %89, %85 ]
  %87 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 36864), i64 %86
  %88 = getelementptr inbounds nuw i8, ptr %87, i64 16
  store <4 x i32> %84, ptr %87, align 4, !tbaa !6
  store <4 x i32> %84, ptr %88, align 4, !tbaa !6
  %89 = add nuw i64 %86, 8
  %90 = icmp eq i64 %89, 1024
  br i1 %90, label %91, label %85, !llvm.loop !35

91:                                               ; preds = %85
  %92 = insertelement <4 x i32> poison, i32 %0, i64 0
  %93 = shufflevector <4 x i32> %92, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %94

94:                                               ; preds = %94, %91
  %95 = phi i64 [ 0, %91 ], [ %98, %94 ]
  %96 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 40960), i64 %95
  %97 = getelementptr inbounds nuw i8, ptr %96, i64 16
  store <4 x i32> %93, ptr %96, align 4, !tbaa !6
  store <4 x i32> %93, ptr %97, align 4, !tbaa !6
  %98 = add nuw i64 %95, 8
  %99 = icmp eq i64 %98, 1024
  br i1 %99, label %100, label %94, !llvm.loop !36

100:                                              ; preds = %94
  %101 = insertelement <4 x i32> poison, i32 %0, i64 0
  %102 = shufflevector <4 x i32> %101, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %103

103:                                              ; preds = %103, %100
  %104 = phi i64 [ 0, %100 ], [ %107, %103 ]
  %105 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 45056), i64 %104
  %106 = getelementptr inbounds nuw i8, ptr %105, i64 16
  store <4 x i32> %102, ptr %105, align 4, !tbaa !6
  store <4 x i32> %102, ptr %106, align 4, !tbaa !6
  %107 = add nuw i64 %104, 8
  %108 = icmp eq i64 %107, 1024
  br i1 %108, label %109, label %103, !llvm.loop !37

109:                                              ; preds = %103
  %110 = insertelement <4 x i32> poison, i32 %0, i64 0
  %111 = shufflevector <4 x i32> %110, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %112

112:                                              ; preds = %112, %109
  %113 = phi i64 [ 0, %109 ], [ %116, %112 ]
  %114 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 49152), i64 %113
  %115 = getelementptr inbounds nuw i8, ptr %114, i64 16
  store <4 x i32> %111, ptr %114, align 4, !tbaa !6
  store <4 x i32> %111, ptr %115, align 4, !tbaa !6
  %116 = add nuw i64 %113, 8
  %117 = icmp eq i64 %116, 1024
  br i1 %117, label %118, label %112, !llvm.loop !38

118:                                              ; preds = %112
  %119 = insertelement <4 x i32> poison, i32 %0, i64 0
  %120 = shufflevector <4 x i32> %119, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %121

121:                                              ; preds = %121, %118
  %122 = phi i64 [ 0, %118 ], [ %125, %121 ]
  %123 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 53248), i64 %122
  %124 = getelementptr inbounds nuw i8, ptr %123, i64 16
  store <4 x i32> %120, ptr %123, align 4, !tbaa !6
  store <4 x i32> %120, ptr %124, align 4, !tbaa !6
  %125 = add nuw i64 %122, 8
  %126 = icmp eq i64 %125, 1024
  br i1 %126, label %127, label %121, !llvm.loop !39

127:                                              ; preds = %121
  %128 = insertelement <4 x i32> poison, i32 %0, i64 0
  %129 = shufflevector <4 x i32> %128, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %130

130:                                              ; preds = %130, %127
  %131 = phi i64 [ 0, %127 ], [ %134, %130 ]
  %132 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 57344), i64 %131
  %133 = getelementptr inbounds nuw i8, ptr %132, i64 16
  store <4 x i32> %129, ptr %132, align 4, !tbaa !6
  store <4 x i32> %129, ptr %133, align 4, !tbaa !6
  %134 = add nuw i64 %131, 8
  %135 = icmp eq i64 %134, 1024
  br i1 %135, label %136, label %130, !llvm.loop !40

136:                                              ; preds = %130
  %137 = insertelement <4 x i32> poison, i32 %0, i64 0
  %138 = shufflevector <4 x i32> %137, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %139

139:                                              ; preds = %139, %136
  %140 = phi i64 [ 0, %136 ], [ %143, %139 ]
  %141 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 61440), i64 %140
  %142 = getelementptr inbounds nuw i8, ptr %141, i64 16
  store <4 x i32> %138, ptr %141, align 4, !tbaa !6
  store <4 x i32> %138, ptr %142, align 4, !tbaa !6
  %143 = add nuw i64 %140, 8
  %144 = icmp eq i64 %143, 1024
  br i1 %144, label %145, label %139, !llvm.loop !41

145:                                              ; preds = %139
  %146 = insertelement <4 x i32> poison, i32 %0, i64 0
  %147 = shufflevector <4 x i32> %146, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %148

148:                                              ; preds = %148, %145
  %149 = phi i64 [ 0, %145 ], [ %152, %148 ]
  %150 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 65536), i64 %149
  %151 = getelementptr inbounds nuw i8, ptr %150, i64 16
  store <4 x i32> %147, ptr %150, align 4, !tbaa !6
  store <4 x i32> %147, ptr %151, align 4, !tbaa !6
  %152 = add nuw i64 %149, 8
  %153 = icmp eq i64 %152, 1024
  br i1 %153, label %154, label %148, !llvm.loop !42

154:                                              ; preds = %148
  %155 = insertelement <4 x i32> poison, i32 %0, i64 0
  %156 = shufflevector <4 x i32> %155, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %157

157:                                              ; preds = %157, %154
  %158 = phi i64 [ 0, %154 ], [ %161, %157 ]
  %159 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 69632), i64 %158
  %160 = getelementptr inbounds nuw i8, ptr %159, i64 16
  store <4 x i32> %156, ptr %159, align 4, !tbaa !6
  store <4 x i32> %156, ptr %160, align 4, !tbaa !6
  %161 = add nuw i64 %158, 8
  %162 = icmp eq i64 %161, 1024
  br i1 %162, label %163, label %157, !llvm.loop !43

163:                                              ; preds = %157
  %164 = insertelement <4 x i32> poison, i32 %0, i64 0
  %165 = shufflevector <4 x i32> %164, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %166

166:                                              ; preds = %166, %163
  %167 = phi i64 [ 0, %163 ], [ %170, %166 ]
  %168 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 73728), i64 %167
  %169 = getelementptr inbounds nuw i8, ptr %168, i64 16
  store <4 x i32> %165, ptr %168, align 4, !tbaa !6
  store <4 x i32> %165, ptr %169, align 4, !tbaa !6
  %170 = add nuw i64 %167, 8
  %171 = icmp eq i64 %170, 1024
  br i1 %171, label %172, label %166, !llvm.loop !44

172:                                              ; preds = %166
  %173 = insertelement <4 x i32> poison, i32 %0, i64 0
  %174 = shufflevector <4 x i32> %173, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %175

175:                                              ; preds = %175, %172
  %176 = phi i64 [ 0, %172 ], [ %179, %175 ]
  %177 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 77824), i64 %176
  %178 = getelementptr inbounds nuw i8, ptr %177, i64 16
  store <4 x i32> %174, ptr %177, align 4, !tbaa !6
  store <4 x i32> %174, ptr %178, align 4, !tbaa !6
  %179 = add nuw i64 %176, 8
  %180 = icmp eq i64 %179, 1024
  br i1 %180, label %181, label %175, !llvm.loop !45

181:                                              ; preds = %175
  %182 = insertelement <4 x i32> poison, i32 %0, i64 0
  %183 = shufflevector <4 x i32> %182, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %184

184:                                              ; preds = %184, %181
  %185 = phi i64 [ 0, %181 ], [ %188, %184 ]
  %186 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 81920), i64 %185
  %187 = getelementptr inbounds nuw i8, ptr %186, i64 16
  store <4 x i32> %183, ptr %186, align 4, !tbaa !6
  store <4 x i32> %183, ptr %187, align 4, !tbaa !6
  %188 = add nuw i64 %185, 8
  %189 = icmp eq i64 %188, 1024
  br i1 %189, label %190, label %184, !llvm.loop !46

190:                                              ; preds = %184
  %191 = insertelement <4 x i32> poison, i32 %0, i64 0
  %192 = shufflevector <4 x i32> %191, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %193

193:                                              ; preds = %193, %190
  %194 = phi i64 [ 0, %190 ], [ %197, %193 ]
  %195 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 86016), i64 %194
  %196 = getelementptr inbounds nuw i8, ptr %195, i64 16
  store <4 x i32> %192, ptr %195, align 4, !tbaa !6
  store <4 x i32> %192, ptr %196, align 4, !tbaa !6
  %197 = add nuw i64 %194, 8
  %198 = icmp eq i64 %197, 1024
  br i1 %198, label %199, label %193, !llvm.loop !47

199:                                              ; preds = %193
  %200 = insertelement <4 x i32> poison, i32 %0, i64 0
  %201 = shufflevector <4 x i32> %200, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %202

202:                                              ; preds = %202, %199
  %203 = phi i64 [ 0, %199 ], [ %206, %202 ]
  %204 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 90112), i64 %203
  %205 = getelementptr inbounds nuw i8, ptr %204, i64 16
  store <4 x i32> %201, ptr %204, align 4, !tbaa !6
  store <4 x i32> %201, ptr %205, align 4, !tbaa !6
  %206 = add nuw i64 %203, 8
  %207 = icmp eq i64 %206, 1024
  br i1 %207, label %208, label %202, !llvm.loop !48

208:                                              ; preds = %202
  %209 = insertelement <4 x i32> poison, i32 %0, i64 0
  %210 = shufflevector <4 x i32> %209, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %211

211:                                              ; preds = %211, %208
  %212 = phi i64 [ 0, %208 ], [ %215, %211 ]
  %213 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 94208), i64 %212
  %214 = getelementptr inbounds nuw i8, ptr %213, i64 16
  store <4 x i32> %210, ptr %213, align 4, !tbaa !6
  store <4 x i32> %210, ptr %214, align 4, !tbaa !6
  %215 = add nuw i64 %212, 8
  %216 = icmp eq i64 %215, 1024
  br i1 %216, label %217, label %211, !llvm.loop !49

217:                                              ; preds = %211
  %218 = insertelement <4 x i32> poison, i32 %0, i64 0
  %219 = shufflevector <4 x i32> %218, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %220

220:                                              ; preds = %220, %217
  %221 = phi i64 [ 0, %217 ], [ %224, %220 ]
  %222 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 98304), i64 %221
  %223 = getelementptr inbounds nuw i8, ptr %222, i64 16
  store <4 x i32> %219, ptr %222, align 4, !tbaa !6
  store <4 x i32> %219, ptr %223, align 4, !tbaa !6
  %224 = add nuw i64 %221, 8
  %225 = icmp eq i64 %224, 1024
  br i1 %225, label %226, label %220, !llvm.loop !50

226:                                              ; preds = %220
  %227 = insertelement <4 x i32> poison, i32 %0, i64 0
  %228 = shufflevector <4 x i32> %227, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %229

229:                                              ; preds = %229, %226
  %230 = phi i64 [ 0, %226 ], [ %233, %229 ]
  %231 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 102400), i64 %230
  %232 = getelementptr inbounds nuw i8, ptr %231, i64 16
  store <4 x i32> %228, ptr %231, align 4, !tbaa !6
  store <4 x i32> %228, ptr %232, align 4, !tbaa !6
  %233 = add nuw i64 %230, 8
  %234 = icmp eq i64 %233, 1024
  br i1 %234, label %235, label %229, !llvm.loop !51

235:                                              ; preds = %229
  %236 = insertelement <4 x i32> poison, i32 %0, i64 0
  %237 = shufflevector <4 x i32> %236, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %238

238:                                              ; preds = %238, %235
  %239 = phi i64 [ 0, %235 ], [ %242, %238 ]
  %240 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 106496), i64 %239
  %241 = getelementptr inbounds nuw i8, ptr %240, i64 16
  store <4 x i32> %237, ptr %240, align 4, !tbaa !6
  store <4 x i32> %237, ptr %241, align 4, !tbaa !6
  %242 = add nuw i64 %239, 8
  %243 = icmp eq i64 %242, 1024
  br i1 %243, label %244, label %238, !llvm.loop !52

244:                                              ; preds = %238
  %245 = insertelement <4 x i32> poison, i32 %0, i64 0
  %246 = shufflevector <4 x i32> %245, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %247

247:                                              ; preds = %247, %244
  %248 = phi i64 [ 0, %244 ], [ %251, %247 ]
  %249 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 110592), i64 %248
  %250 = getelementptr inbounds nuw i8, ptr %249, i64 16
  store <4 x i32> %246, ptr %249, align 4, !tbaa !6
  store <4 x i32> %246, ptr %250, align 4, !tbaa !6
  %251 = add nuw i64 %248, 8
  %252 = icmp eq i64 %251, 1024
  br i1 %252, label %253, label %247, !llvm.loop !53

253:                                              ; preds = %247
  %254 = insertelement <4 x i32> poison, i32 %0, i64 0
  %255 = shufflevector <4 x i32> %254, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %256

256:                                              ; preds = %256, %253
  %257 = phi i64 [ 0, %253 ], [ %260, %256 ]
  %258 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 114688), i64 %257
  %259 = getelementptr inbounds nuw i8, ptr %258, i64 16
  store <4 x i32> %255, ptr %258, align 4, !tbaa !6
  store <4 x i32> %255, ptr %259, align 4, !tbaa !6
  %260 = add nuw i64 %257, 8
  %261 = icmp eq i64 %260, 1024
  br i1 %261, label %262, label %256, !llvm.loop !54

262:                                              ; preds = %256
  %263 = insertelement <4 x i32> poison, i32 %0, i64 0
  %264 = shufflevector <4 x i32> %263, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %265

265:                                              ; preds = %265, %262
  %266 = phi i64 [ 0, %262 ], [ %269, %265 ]
  %267 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 118784), i64 %266
  %268 = getelementptr inbounds nuw i8, ptr %267, i64 16
  store <4 x i32> %264, ptr %267, align 4, !tbaa !6
  store <4 x i32> %264, ptr %268, align 4, !tbaa !6
  %269 = add nuw i64 %266, 8
  %270 = icmp eq i64 %269, 1024
  br i1 %270, label %271, label %265, !llvm.loop !55

271:                                              ; preds = %265
  %272 = insertelement <4 x i32> poison, i32 %0, i64 0
  %273 = shufflevector <4 x i32> %272, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %274

274:                                              ; preds = %274, %271
  %275 = phi i64 [ 0, %271 ], [ %278, %274 ]
  %276 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 122880), i64 %275
  %277 = getelementptr inbounds nuw i8, ptr %276, i64 16
  store <4 x i32> %273, ptr %276, align 4, !tbaa !6
  store <4 x i32> %273, ptr %277, align 4, !tbaa !6
  %278 = add nuw i64 %275, 8
  %279 = icmp eq i64 %278, 1024
  br i1 %279, label %280, label %274, !llvm.loop !56

280:                                              ; preds = %274
  %281 = insertelement <4 x i32> poison, i32 %0, i64 0
  %282 = shufflevector <4 x i32> %281, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %283

283:                                              ; preds = %283, %280
  %284 = phi i64 [ 0, %280 ], [ %287, %283 ]
  %285 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @G, i64 126976), i64 %284
  %286 = getelementptr inbounds nuw i8, ptr %285, i64 16
  store <4 x i32> %282, ptr %285, align 4, !tbaa !6
  store <4 x i32> %282, ptr %286, align 4, !tbaa !6
  %287 = add nuw i64 %284, 8
  %288 = icmp eq i64 %287, 1024
  br i1 %288, label %289, label %283, !llvm.loop !57

289:                                              ; preds = %283
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(read, argmem: write, inaccessiblemem: none) uwtable
define dso_local void @_Z8example9Pj(ptr noundef writeonly captures(none) %0) local_unnamed_addr #7 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi i64 [ 0, %1 ], [ %18, %2 ]
  %4 = phi <4 x i32> [ zeroinitializer, %1 ], [ %16, %2 ]
  %5 = phi <4 x i32> [ zeroinitializer, %1 ], [ %17, %2 ]
  %6 = getelementptr inbounds nuw i32, ptr @ub, i64 %3
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %8 = load <4 x i32>, ptr %6, align 4, !tbaa !6
  %9 = load <4 x i32>, ptr %7, align 4, !tbaa !6
  %10 = getelementptr inbounds nuw i32, ptr @uc, i64 %3
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %12 = load <4 x i32>, ptr %10, align 4, !tbaa !6
  %13 = load <4 x i32>, ptr %11, align 4, !tbaa !6
  %14 = add <4 x i32> %8, %4
  %15 = add <4 x i32> %9, %5
  %16 = sub <4 x i32> %14, %12
  %17 = sub <4 x i32> %15, %13
  %18 = add nuw i64 %3, 8
  %19 = icmp eq i64 %18, 1024
  br i1 %19, label %20, label %2, !llvm.loop !58

20:                                               ; preds = %2
  %21 = add <4 x i32> %17, %16
  %22 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %21)
  store i32 %22, ptr %0, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z10example10aPsS_S_PiS0_S0_(ptr noalias noundef writeonly captures(none) %0, ptr noalias noundef readonly captures(none) %1, ptr noalias noundef readonly captures(none) %2, ptr noalias noundef writeonly captures(none) %3, ptr noalias noundef readonly captures(none) %4, ptr noalias noundef readonly captures(none) %5) local_unnamed_addr #4 {
  br label %7

7:                                                ; preds = %7, %6
  %8 = phi i64 [ 0, %6 ], [ %21, %7 ]
  %9 = getelementptr inbounds nuw i32, ptr %4, i64 %8
  %10 = load <8 x i32>, ptr %9, align 4, !tbaa !6
  %11 = getelementptr inbounds nuw i32, ptr %5, i64 %8
  %12 = load <8 x i32>, ptr %11, align 4, !tbaa !6
  %13 = add nsw <8 x i32> %12, %10
  %14 = getelementptr inbounds nuw i32, ptr %3, i64 %8
  store <8 x i32> %13, ptr %14, align 4, !tbaa !6
  %15 = getelementptr inbounds nuw i16, ptr %1, i64 %8
  %16 = load <8 x i16>, ptr %15, align 2, !tbaa !59
  %17 = getelementptr inbounds nuw i16, ptr %2, i64 %8
  %18 = load <8 x i16>, ptr %17, align 2, !tbaa !59
  %19 = add <8 x i16> %18, %16
  %20 = getelementptr inbounds nuw i16, ptr %0, i64 %8
  store <8 x i16> %19, ptr %20, align 2, !tbaa !59
  %21 = add nuw i64 %8, 8
  %22 = icmp eq i64 %21, 1024
  br i1 %22, label %23, label %7, !llvm.loop !61

23:                                               ; preds = %7
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z10example10bPsS_S_PiS0_S0_(ptr noalias readnone captures(none) %0, ptr noalias noundef readonly captures(none) %1, ptr noalias readnone captures(none) %2, ptr noalias noundef writeonly captures(none) %3, ptr noalias readnone captures(none) %4, ptr noalias readnone captures(none) %5) local_unnamed_addr #4 {
  br label %7

7:                                                ; preds = %7, %6
  %8 = phi i64 [ 0, %6 ], [ %17, %7 ]
  %9 = getelementptr inbounds nuw i16, ptr %1, i64 %8
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load <8 x i16>, ptr %9, align 2, !tbaa !59
  %12 = load <8 x i16>, ptr %10, align 2, !tbaa !59
  %13 = sext <8 x i16> %11 to <8 x i32>
  %14 = sext <8 x i16> %12 to <8 x i32>
  %15 = getelementptr inbounds nuw i32, ptr %3, i64 %8
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 32
  store <8 x i32> %13, ptr %15, align 4, !tbaa !6
  store <8 x i32> %14, ptr %16, align 4, !tbaa !6
  %17 = add nuw i64 %8, 16
  %18 = icmp eq i64 %17, 1024
  br i1 %18, label %19, label %7, !llvm.loop !62

19:                                               ; preds = %7
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z9example11v() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %23, %1 ]
  %3 = shl nuw nsw i64 %2, 1
  %4 = or disjoint i64 %3, 1
  %5 = getelementptr inbounds nuw i32, ptr @b, i64 %4
  %6 = getelementptr inbounds i8, ptr %5, i64 -4
  %7 = load <8 x i32>, ptr %6, align 16, !tbaa !6
  %8 = shufflevector <8 x i32> %7, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %9 = shufflevector <8 x i32> %7, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %10 = getelementptr inbounds nuw i32, ptr @c, i64 %4
  %11 = getelementptr inbounds i8, ptr %10, i64 -4
  %12 = load <8 x i32>, ptr %11, align 16, !tbaa !6
  %13 = shufflevector <8 x i32> %12, <8 x i32> poison, <4 x i32> <i32 0, i32 2, i32 4, i32 6>
  %14 = shufflevector <8 x i32> %12, <8 x i32> poison, <4 x i32> <i32 1, i32 3, i32 5, i32 7>
  %15 = mul nsw <4 x i32> %14, %9
  %16 = mul nsw <4 x i32> %13, %8
  %17 = sub nsw <4 x i32> %15, %16
  %18 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  store <4 x i32> %17, ptr %18, align 16, !tbaa !6
  %19 = mul nsw <4 x i32> %8, %14
  %20 = mul nsw <4 x i32> %13, %9
  %21 = add nsw <4 x i32> %20, %19
  %22 = getelementptr inbounds nuw i32, ptr @d, i64 %2
  store <4 x i32> %21, ptr %22, align 16, !tbaa !6
  %23 = add nuw i64 %2, 4
  %24 = icmp eq i64 %23, 512
  br i1 %24, label %25, label %1, !llvm.loop !63

25:                                               ; preds = %1
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z9example12v() local_unnamed_addr #2 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %7, %1 ]
  %3 = phi <4 x i32> [ <i32 0, i32 1, i32 2, i32 3>, %0 ], [ %8, %1 ]
  %4 = add <4 x i32> %3, splat (i32 4)
  %5 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  %6 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store <4 x i32> %3, ptr %5, align 16, !tbaa !6
  store <4 x i32> %4, ptr %6, align 16, !tbaa !6
  %7 = add nuw i64 %2, 8
  %8 = add <4 x i32> %3, splat (i32 8)
  %9 = icmp eq i64 %7, 1024
  br i1 %9, label %10, label %1, !llvm.loop !64

10:                                               ; preds = %1
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @_Z9example13PPiS0_S_(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) local_unnamed_addr #8 {
  br label %4

4:                                                ; preds = %3, %30
  %5 = phi i64 [ 0, %3 ], [ %33, %30 ]
  %6 = getelementptr inbounds nuw ptr, ptr %0, i64 %5
  %7 = load ptr, ptr %6, align 8, !tbaa !65
  %8 = getelementptr inbounds nuw ptr, ptr %1, i64 %5
  %9 = load ptr, ptr %8, align 8, !tbaa !65
  br label %10

10:                                               ; preds = %10, %4
  %11 = phi i64 [ 0, %4 ], [ %28, %10 ]
  %12 = phi i32 [ 0, %4 ], [ %26, %10 ]
  %13 = phi i32 [ 0, %4 ], [ %27, %10 ]
  %14 = shl i64 %11, 3
  %15 = or disjoint i64 %14, 8
  %16 = getelementptr inbounds nuw i32, ptr %7, i64 %14
  %17 = getelementptr inbounds nuw i32, ptr %7, i64 %15
  %18 = load i32, ptr %16, align 4, !tbaa !6
  %19 = load i32, ptr %17, align 4, !tbaa !6
  %20 = getelementptr inbounds nuw i32, ptr %9, i64 %14
  %21 = getelementptr inbounds nuw i32, ptr %9, i64 %15
  %22 = load i32, ptr %20, align 4, !tbaa !6
  %23 = load i32, ptr %21, align 4, !tbaa !6
  %24 = add i32 %18, %12
  %25 = add i32 %19, %13
  %26 = sub i32 %24, %22
  %27 = sub i32 %25, %23
  %28 = add nuw i64 %11, 2
  %29 = icmp eq i64 %28, 128
  br i1 %29, label %30, label %10, !llvm.loop !68

30:                                               ; preds = %10
  %31 = add i32 %27, %26
  %32 = getelementptr inbounds nuw i32, ptr %2, i64 %5
  store i32 %31, ptr %32, align 4, !tbaa !6
  %33 = add nuw nsw i64 %5, 1
  %34 = icmp eq i64 %33, 32
  br i1 %34, label %35, label %4, !llvm.loop !69

35:                                               ; preds = %30
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @_Z9example14PPiS0_S_(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2) local_unnamed_addr #8 {
  br label %4

4:                                                ; preds = %34, %3
  %5 = phi i64 [ 0, %3 ], [ %36, %34 ]
  %6 = phi i32 [ 0, %3 ], [ %35, %34 ]
  br label %7

7:                                                ; preds = %7, %4
  %8 = phi i64 [ 0, %4 ], [ %32, %7 ]
  %9 = phi i32 [ %6, %4 ], [ %30, %7 ]
  %10 = phi i32 [ 0, %4 ], [ %31, %7 ]
  %11 = or disjoint i64 %8, 1
  %12 = getelementptr inbounds nuw ptr, ptr %0, i64 %8
  %13 = getelementptr inbounds nuw ptr, ptr %0, i64 %11
  %14 = load ptr, ptr %12, align 8, !tbaa !65
  %15 = load ptr, ptr %13, align 8, !tbaa !65
  %16 = getelementptr inbounds nuw i32, ptr %14, i64 %5
  %17 = getelementptr inbounds nuw i32, ptr %15, i64 %5
  %18 = load i32, ptr %16, align 4, !tbaa !6
  %19 = load i32, ptr %17, align 4, !tbaa !6
  %20 = getelementptr inbounds nuw ptr, ptr %1, i64 %8
  %21 = getelementptr inbounds nuw ptr, ptr %1, i64 %11
  %22 = load ptr, ptr %20, align 8, !tbaa !65
  %23 = load ptr, ptr %21, align 8, !tbaa !65
  %24 = getelementptr inbounds nuw i32, ptr %22, i64 %5
  %25 = getelementptr inbounds nuw i32, ptr %23, i64 %5
  %26 = load i32, ptr %24, align 4, !tbaa !6
  %27 = load i32, ptr %25, align 4, !tbaa !6
  %28 = mul nsw i32 %26, %18
  %29 = mul nsw i32 %27, %19
  %30 = add i32 %28, %9
  %31 = add i32 %29, %10
  %32 = add nuw i64 %8, 2
  %33 = icmp eq i64 %32, 1024
  br i1 %33, label %34, label %7, !llvm.loop !70

34:                                               ; preds = %7
  %35 = add i32 %31, %30
  %36 = add nuw nsw i64 %5, 1
  %37 = icmp eq i64 %36, 32
  br i1 %37, label %38, label %4, !llvm.loop !71

38:                                               ; preds = %34
  store i32 %35, ptr %2, align 4, !tbaa !6
  br label %39

39:                                               ; preds = %38, %71
  %40 = phi i64 [ 0, %38 ], [ %73, %71 ]
  %41 = phi i32 [ 0, %38 ], [ %72, %71 ]
  br label %42

42:                                               ; preds = %42, %39
  %43 = phi i64 [ 0, %39 ], [ %69, %42 ]
  %44 = phi i32 [ %41, %39 ], [ %67, %42 ]
  %45 = phi i32 [ 0, %39 ], [ %68, %42 ]
  %46 = getelementptr inbounds nuw ptr, ptr %0, i64 %43
  %47 = getelementptr inbounds nuw i8, ptr %46, i64 8
  %48 = getelementptr inbounds nuw ptr, ptr %0, i64 %43
  %49 = getelementptr inbounds nuw i8, ptr %48, i64 16
  %50 = load ptr, ptr %47, align 8, !tbaa !65
  %51 = load ptr, ptr %49, align 8, !tbaa !65
  %52 = getelementptr inbounds nuw i32, ptr %50, i64 %40
  %53 = getelementptr inbounds nuw i32, ptr %51, i64 %40
  %54 = load i32, ptr %52, align 4, !tbaa !6
  %55 = load i32, ptr %53, align 4, !tbaa !6
  %56 = getelementptr inbounds nuw ptr, ptr %1, i64 %43
  %57 = getelementptr inbounds nuw ptr, ptr %1, i64 %43
  %58 = getelementptr inbounds nuw i8, ptr %57, i64 8
  %59 = load ptr, ptr %56, align 8, !tbaa !65
  %60 = load ptr, ptr %58, align 8, !tbaa !65
  %61 = getelementptr inbounds nuw i32, ptr %59, i64 %40
  %62 = getelementptr inbounds nuw i32, ptr %60, i64 %40
  %63 = load i32, ptr %61, align 4, !tbaa !6
  %64 = load i32, ptr %62, align 4, !tbaa !6
  %65 = mul nsw i32 %63, %54
  %66 = mul nsw i32 %64, %55
  %67 = add i32 %65, %44
  %68 = add i32 %66, %45
  %69 = add nuw i64 %43, 2
  %70 = icmp eq i64 %69, 1024
  br i1 %70, label %71, label %42, !llvm.loop !72

71:                                               ; preds = %42
  %72 = add i32 %68, %67
  %73 = add nuw nsw i64 %40, 1
  %74 = icmp eq i64 %73, 32
  br i1 %74, label %75, label %39, !llvm.loop !71

75:                                               ; preds = %71
  %76 = getelementptr inbounds nuw i8, ptr %2, i64 4
  store i32 %72, ptr %76, align 4, !tbaa !6
  br label %77

77:                                               ; preds = %75, %109
  %78 = phi i64 [ 0, %75 ], [ %111, %109 ]
  %79 = phi i32 [ 0, %75 ], [ %110, %109 ]
  br label %80

80:                                               ; preds = %80, %77
  %81 = phi i64 [ 0, %77 ], [ %107, %80 ]
  %82 = phi i32 [ %79, %77 ], [ %105, %80 ]
  %83 = phi i32 [ 0, %77 ], [ %106, %80 ]
  %84 = or disjoint i64 %81, 1
  %85 = getelementptr inbounds nuw ptr, ptr %0, i64 %81
  %86 = getelementptr inbounds nuw ptr, ptr %0, i64 %84
  %87 = getelementptr inbounds nuw i8, ptr %85, i64 16
  %88 = getelementptr inbounds nuw i8, ptr %86, i64 16
  %89 = load ptr, ptr %87, align 8, !tbaa !65
  %90 = load ptr, ptr %88, align 8, !tbaa !65
  %91 = getelementptr inbounds nuw i32, ptr %89, i64 %78
  %92 = getelementptr inbounds nuw i32, ptr %90, i64 %78
  %93 = load i32, ptr %91, align 4, !tbaa !6
  %94 = load i32, ptr %92, align 4, !tbaa !6
  %95 = getelementptr inbounds nuw ptr, ptr %1, i64 %81
  %96 = getelementptr inbounds nuw ptr, ptr %1, i64 %84
  %97 = load ptr, ptr %95, align 8, !tbaa !65
  %98 = load ptr, ptr %96, align 8, !tbaa !65
  %99 = getelementptr inbounds nuw i32, ptr %97, i64 %78
  %100 = getelementptr inbounds nuw i32, ptr %98, i64 %78
  %101 = load i32, ptr %99, align 4, !tbaa !6
  %102 = load i32, ptr %100, align 4, !tbaa !6
  %103 = mul nsw i32 %101, %93
  %104 = mul nsw i32 %102, %94
  %105 = add i32 %103, %82
  %106 = add i32 %104, %83
  %107 = add nuw i64 %81, 2
  %108 = icmp eq i64 %107, 1024
  br i1 %108, label %109, label %80, !llvm.loop !73

109:                                              ; preds = %80
  %110 = add i32 %106, %105
  %111 = add nuw nsw i64 %78, 1
  %112 = icmp eq i64 %111, 32
  br i1 %112, label %113, label %77, !llvm.loop !71

113:                                              ; preds = %109
  %114 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i32 %110, ptr %114, align 4, !tbaa !6
  br label %115

115:                                              ; preds = %113, %147
  %116 = phi i64 [ 0, %113 ], [ %149, %147 ]
  %117 = phi i32 [ 0, %113 ], [ %148, %147 ]
  br label %118

118:                                              ; preds = %118, %115
  %119 = phi i64 [ 0, %115 ], [ %145, %118 ]
  %120 = phi i32 [ %117, %115 ], [ %143, %118 ]
  %121 = phi i32 [ 0, %115 ], [ %144, %118 ]
  %122 = or disjoint i64 %119, 1
  %123 = getelementptr inbounds nuw ptr, ptr %0, i64 %119
  %124 = getelementptr inbounds nuw ptr, ptr %0, i64 %122
  %125 = getelementptr inbounds nuw i8, ptr %123, i64 24
  %126 = getelementptr inbounds nuw i8, ptr %124, i64 24
  %127 = load ptr, ptr %125, align 8, !tbaa !65
  %128 = load ptr, ptr %126, align 8, !tbaa !65
  %129 = getelementptr inbounds nuw i32, ptr %127, i64 %116
  %130 = getelementptr inbounds nuw i32, ptr %128, i64 %116
  %131 = load i32, ptr %129, align 4, !tbaa !6
  %132 = load i32, ptr %130, align 4, !tbaa !6
  %133 = getelementptr inbounds nuw ptr, ptr %1, i64 %119
  %134 = getelementptr inbounds nuw ptr, ptr %1, i64 %122
  %135 = load ptr, ptr %133, align 8, !tbaa !65
  %136 = load ptr, ptr %134, align 8, !tbaa !65
  %137 = getelementptr inbounds nuw i32, ptr %135, i64 %116
  %138 = getelementptr inbounds nuw i32, ptr %136, i64 %116
  %139 = load i32, ptr %137, align 4, !tbaa !6
  %140 = load i32, ptr %138, align 4, !tbaa !6
  %141 = mul nsw i32 %139, %131
  %142 = mul nsw i32 %140, %132
  %143 = add i32 %141, %120
  %144 = add i32 %142, %121
  %145 = add nuw i64 %119, 2
  %146 = icmp eq i64 %145, 1024
  br i1 %146, label %147, label %118, !llvm.loop !74

147:                                              ; preds = %118
  %148 = add i32 %144, %143
  %149 = add nuw nsw i64 %116, 1
  %150 = icmp eq i64 %149, 32
  br i1 %150, label %151, label %115, !llvm.loop !71

151:                                              ; preds = %147
  %152 = getelementptr inbounds nuw i8, ptr %2, i64 12
  store i32 %148, ptr %152, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z9example21Pii(ptr noundef captures(none) %0, i32 noundef %1) local_unnamed_addr #4 {
  %3 = icmp sgt i32 %1, 0
  br i1 %3, label %4, label %42

4:                                                ; preds = %2
  %5 = zext nneg i32 %1 to i64
  %6 = icmp ult i32 %1, 8
  br i1 %6, label %31, label %7

7:                                                ; preds = %4
  %8 = and i64 %5, 2147483640
  %9 = and i64 %5, 7
  %10 = getelementptr i32, ptr %0, i64 %5
  br label %11

11:                                               ; preds = %11, %7
  %12 = phi i64 [ 0, %7 ], [ %25, %11 ]
  %13 = phi <4 x i32> [ zeroinitializer, %7 ], [ %23, %11 ]
  %14 = phi <4 x i32> [ zeroinitializer, %7 ], [ %24, %11 ]
  %15 = xor i64 %12, -1
  %16 = getelementptr i32, ptr %10, i64 %15
  %17 = getelementptr inbounds i8, ptr %16, i64 -12
  %18 = getelementptr inbounds i8, ptr %16, i64 -28
  %19 = load <4 x i32>, ptr %17, align 4, !tbaa !6
  %20 = shufflevector <4 x i32> %19, <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %21 = load <4 x i32>, ptr %18, align 4, !tbaa !6
  %22 = shufflevector <4 x i32> %21, <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
  %23 = add <4 x i32> %20, %13
  %24 = add <4 x i32> %22, %14
  %25 = add nuw i64 %12, 8
  %26 = icmp eq i64 %25, %8
  br i1 %26, label %27, label %11, !llvm.loop !75

27:                                               ; preds = %11
  %28 = add <4 x i32> %24, %23
  %29 = tail call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %28)
  %30 = icmp eq i64 %8, %5
  br i1 %30, label %42, label %31

31:                                               ; preds = %4, %27
  %32 = phi i64 [ %5, %4 ], [ %9, %27 ]
  %33 = phi i32 [ 0, %4 ], [ %29, %27 ]
  br label %34

34:                                               ; preds = %31, %34
  %35 = phi i64 [ %37, %34 ], [ %32, %31 ]
  %36 = phi i32 [ %40, %34 ], [ %33, %31 ]
  %37 = add nsw i64 %35, -1
  %38 = getelementptr inbounds nuw i32, ptr %0, i64 %37
  %39 = load i32, ptr %38, align 4, !tbaa !6
  %40 = add nsw i32 %39, %36
  %41 = icmp samesign ugt i64 %35, 1
  br i1 %41, label %34, label %42, !llvm.loop !76

42:                                               ; preds = %34, %27, %2
  %43 = phi i32 [ 0, %2 ], [ %29, %27 ], [ %40, %34 ]
  store i32 %43, ptr %0, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @_Z9example23PtPj(ptr noundef readonly captures(none) %0, ptr noundef writeonly captures(none) %1) local_unnamed_addr #4 {
  %3 = getelementptr i8, ptr %0, i64 16
  %4 = load <8 x i16>, ptr %0, align 2, !tbaa !59
  %5 = load <8 x i16>, ptr %3, align 2, !tbaa !59
  %6 = zext <8 x i16> %4 to <8 x i32>
  %7 = zext <8 x i16> %5 to <8 x i32>
  %8 = shl nuw nsw <8 x i32> %6, splat (i32 7)
  %9 = shl nuw nsw <8 x i32> %7, splat (i32 7)
  %10 = getelementptr i8, ptr %1, i64 32
  store <8 x i32> %8, ptr %1, align 4, !tbaa !6
  store <8 x i32> %9, ptr %10, align 4, !tbaa !6
  %11 = getelementptr i8, ptr %1, i64 64
  %12 = getelementptr i8, ptr %0, i64 32
  %13 = getelementptr i8, ptr %0, i64 48
  %14 = load <8 x i16>, ptr %12, align 2, !tbaa !59
  %15 = load <8 x i16>, ptr %13, align 2, !tbaa !59
  %16 = zext <8 x i16> %14 to <8 x i32>
  %17 = zext <8 x i16> %15 to <8 x i32>
  %18 = shl nuw nsw <8 x i32> %16, splat (i32 7)
  %19 = shl nuw nsw <8 x i32> %17, splat (i32 7)
  %20 = getelementptr i8, ptr %1, i64 96
  store <8 x i32> %18, ptr %11, align 4, !tbaa !6
  store <8 x i32> %19, ptr %20, align 4, !tbaa !6
  %21 = getelementptr i8, ptr %1, i64 128
  %22 = getelementptr i8, ptr %0, i64 64
  %23 = getelementptr i8, ptr %0, i64 80
  %24 = load <8 x i16>, ptr %22, align 2, !tbaa !59
  %25 = load <8 x i16>, ptr %23, align 2, !tbaa !59
  %26 = zext <8 x i16> %24 to <8 x i32>
  %27 = zext <8 x i16> %25 to <8 x i32>
  %28 = shl nuw nsw <8 x i32> %26, splat (i32 7)
  %29 = shl nuw nsw <8 x i32> %27, splat (i32 7)
  %30 = getelementptr i8, ptr %1, i64 160
  store <8 x i32> %28, ptr %21, align 4, !tbaa !6
  store <8 x i32> %29, ptr %30, align 4, !tbaa !6
  %31 = getelementptr i8, ptr %1, i64 192
  %32 = getelementptr i8, ptr %0, i64 96
  %33 = getelementptr i8, ptr %0, i64 112
  %34 = load <8 x i16>, ptr %32, align 2, !tbaa !59
  %35 = load <8 x i16>, ptr %33, align 2, !tbaa !59
  %36 = zext <8 x i16> %34 to <8 x i32>
  %37 = zext <8 x i16> %35 to <8 x i32>
  %38 = shl nuw nsw <8 x i32> %36, splat (i32 7)
  %39 = shl nuw nsw <8 x i32> %37, splat (i32 7)
  %40 = getelementptr i8, ptr %1, i64 224
  store <8 x i32> %38, ptr %31, align 4, !tbaa !6
  store <8 x i32> %39, ptr %40, align 4, !tbaa !6
  %41 = getelementptr i8, ptr %1, i64 256
  %42 = getelementptr i8, ptr %0, i64 128
  %43 = getelementptr i8, ptr %0, i64 144
  %44 = load <8 x i16>, ptr %42, align 2, !tbaa !59
  %45 = load <8 x i16>, ptr %43, align 2, !tbaa !59
  %46 = zext <8 x i16> %44 to <8 x i32>
  %47 = zext <8 x i16> %45 to <8 x i32>
  %48 = shl nuw nsw <8 x i32> %46, splat (i32 7)
  %49 = shl nuw nsw <8 x i32> %47, splat (i32 7)
  %50 = getelementptr i8, ptr %1, i64 288
  store <8 x i32> %48, ptr %41, align 4, !tbaa !6
  store <8 x i32> %49, ptr %50, align 4, !tbaa !6
  %51 = getelementptr i8, ptr %1, i64 320
  %52 = getelementptr i8, ptr %0, i64 160
  %53 = getelementptr i8, ptr %0, i64 176
  %54 = load <8 x i16>, ptr %52, align 2, !tbaa !59
  %55 = load <8 x i16>, ptr %53, align 2, !tbaa !59
  %56 = zext <8 x i16> %54 to <8 x i32>
  %57 = zext <8 x i16> %55 to <8 x i32>
  %58 = shl nuw nsw <8 x i32> %56, splat (i32 7)
  %59 = shl nuw nsw <8 x i32> %57, splat (i32 7)
  %60 = getelementptr i8, ptr %1, i64 352
  store <8 x i32> %58, ptr %51, align 4, !tbaa !6
  store <8 x i32> %59, ptr %60, align 4, !tbaa !6
  %61 = getelementptr i8, ptr %1, i64 384
  %62 = getelementptr i8, ptr %0, i64 192
  %63 = getelementptr i8, ptr %0, i64 208
  %64 = load <8 x i16>, ptr %62, align 2, !tbaa !59
  %65 = load <8 x i16>, ptr %63, align 2, !tbaa !59
  %66 = zext <8 x i16> %64 to <8 x i32>
  %67 = zext <8 x i16> %65 to <8 x i32>
  %68 = shl nuw nsw <8 x i32> %66, splat (i32 7)
  %69 = shl nuw nsw <8 x i32> %67, splat (i32 7)
  %70 = getelementptr i8, ptr %1, i64 416
  store <8 x i32> %68, ptr %61, align 4, !tbaa !6
  store <8 x i32> %69, ptr %70, align 4, !tbaa !6
  %71 = getelementptr i8, ptr %1, i64 448
  %72 = getelementptr i8, ptr %0, i64 224
  %73 = getelementptr i8, ptr %0, i64 240
  %74 = load <8 x i16>, ptr %72, align 2, !tbaa !59
  %75 = load <8 x i16>, ptr %73, align 2, !tbaa !59
  %76 = zext <8 x i16> %74 to <8 x i32>
  %77 = zext <8 x i16> %75 to <8 x i32>
  %78 = shl nuw nsw <8 x i32> %76, splat (i32 7)
  %79 = shl nuw nsw <8 x i32> %77, splat (i32 7)
  %80 = getelementptr i8, ptr %1, i64 480
  store <8 x i32> %78, ptr %71, align 4, !tbaa !6
  store <8 x i32> %79, ptr %80, align 4, !tbaa !6
  %81 = getelementptr i8, ptr %1, i64 512
  %82 = getelementptr i8, ptr %0, i64 256
  %83 = getelementptr i8, ptr %0, i64 272
  %84 = load <8 x i16>, ptr %82, align 2, !tbaa !59
  %85 = load <8 x i16>, ptr %83, align 2, !tbaa !59
  %86 = zext <8 x i16> %84 to <8 x i32>
  %87 = zext <8 x i16> %85 to <8 x i32>
  %88 = shl nuw nsw <8 x i32> %86, splat (i32 7)
  %89 = shl nuw nsw <8 x i32> %87, splat (i32 7)
  %90 = getelementptr i8, ptr %1, i64 544
  store <8 x i32> %88, ptr %81, align 4, !tbaa !6
  store <8 x i32> %89, ptr %90, align 4, !tbaa !6
  %91 = getelementptr i8, ptr %1, i64 576
  %92 = getelementptr i8, ptr %0, i64 288
  %93 = getelementptr i8, ptr %0, i64 304
  %94 = load <8 x i16>, ptr %92, align 2, !tbaa !59
  %95 = load <8 x i16>, ptr %93, align 2, !tbaa !59
  %96 = zext <8 x i16> %94 to <8 x i32>
  %97 = zext <8 x i16> %95 to <8 x i32>
  %98 = shl nuw nsw <8 x i32> %96, splat (i32 7)
  %99 = shl nuw nsw <8 x i32> %97, splat (i32 7)
  %100 = getelementptr i8, ptr %1, i64 608
  store <8 x i32> %98, ptr %91, align 4, !tbaa !6
  store <8 x i32> %99, ptr %100, align 4, !tbaa !6
  %101 = getelementptr i8, ptr %1, i64 640
  %102 = getelementptr i8, ptr %0, i64 320
  %103 = getelementptr i8, ptr %0, i64 336
  %104 = load <8 x i16>, ptr %102, align 2, !tbaa !59
  %105 = load <8 x i16>, ptr %103, align 2, !tbaa !59
  %106 = zext <8 x i16> %104 to <8 x i32>
  %107 = zext <8 x i16> %105 to <8 x i32>
  %108 = shl nuw nsw <8 x i32> %106, splat (i32 7)
  %109 = shl nuw nsw <8 x i32> %107, splat (i32 7)
  %110 = getelementptr i8, ptr %1, i64 672
  store <8 x i32> %108, ptr %101, align 4, !tbaa !6
  store <8 x i32> %109, ptr %110, align 4, !tbaa !6
  %111 = getelementptr i8, ptr %1, i64 704
  %112 = getelementptr i8, ptr %0, i64 352
  %113 = getelementptr i8, ptr %0, i64 368
  %114 = load <8 x i16>, ptr %112, align 2, !tbaa !59
  %115 = load <8 x i16>, ptr %113, align 2, !tbaa !59
  %116 = zext <8 x i16> %114 to <8 x i32>
  %117 = zext <8 x i16> %115 to <8 x i32>
  %118 = shl nuw nsw <8 x i32> %116, splat (i32 7)
  %119 = shl nuw nsw <8 x i32> %117, splat (i32 7)
  %120 = getelementptr i8, ptr %1, i64 736
  store <8 x i32> %118, ptr %111, align 4, !tbaa !6
  store <8 x i32> %119, ptr %120, align 4, !tbaa !6
  %121 = getelementptr i8, ptr %1, i64 768
  %122 = getelementptr i8, ptr %0, i64 384
  %123 = getelementptr i8, ptr %0, i64 400
  %124 = load <8 x i16>, ptr %122, align 2, !tbaa !59
  %125 = load <8 x i16>, ptr %123, align 2, !tbaa !59
  %126 = zext <8 x i16> %124 to <8 x i32>
  %127 = zext <8 x i16> %125 to <8 x i32>
  %128 = shl nuw nsw <8 x i32> %126, splat (i32 7)
  %129 = shl nuw nsw <8 x i32> %127, splat (i32 7)
  %130 = getelementptr i8, ptr %1, i64 800
  store <8 x i32> %128, ptr %121, align 4, !tbaa !6
  store <8 x i32> %129, ptr %130, align 4, !tbaa !6
  %131 = getelementptr i8, ptr %1, i64 832
  %132 = getelementptr i8, ptr %0, i64 416
  %133 = getelementptr i8, ptr %0, i64 432
  %134 = load <8 x i16>, ptr %132, align 2, !tbaa !59
  %135 = load <8 x i16>, ptr %133, align 2, !tbaa !59
  %136 = zext <8 x i16> %134 to <8 x i32>
  %137 = zext <8 x i16> %135 to <8 x i32>
  %138 = shl nuw nsw <8 x i32> %136, splat (i32 7)
  %139 = shl nuw nsw <8 x i32> %137, splat (i32 7)
  %140 = getelementptr i8, ptr %1, i64 864
  store <8 x i32> %138, ptr %131, align 4, !tbaa !6
  store <8 x i32> %139, ptr %140, align 4, !tbaa !6
  %141 = getelementptr i8, ptr %1, i64 896
  %142 = getelementptr i8, ptr %0, i64 448
  %143 = getelementptr i8, ptr %0, i64 464
  %144 = load <8 x i16>, ptr %142, align 2, !tbaa !59
  %145 = load <8 x i16>, ptr %143, align 2, !tbaa !59
  %146 = zext <8 x i16> %144 to <8 x i32>
  %147 = zext <8 x i16> %145 to <8 x i32>
  %148 = shl nuw nsw <8 x i32> %146, splat (i32 7)
  %149 = shl nuw nsw <8 x i32> %147, splat (i32 7)
  %150 = getelementptr i8, ptr %1, i64 928
  store <8 x i32> %148, ptr %141, align 4, !tbaa !6
  store <8 x i32> %149, ptr %150, align 4, !tbaa !6
  %151 = getelementptr i8, ptr %1, i64 960
  %152 = getelementptr i8, ptr %0, i64 480
  %153 = getelementptr i8, ptr %0, i64 496
  %154 = load <8 x i16>, ptr %152, align 2, !tbaa !59
  %155 = load <8 x i16>, ptr %153, align 2, !tbaa !59
  %156 = zext <8 x i16> %154 to <8 x i32>
  %157 = zext <8 x i16> %155 to <8 x i32>
  %158 = shl nuw nsw <8 x i32> %156, splat (i32 7)
  %159 = shl nuw nsw <8 x i32> %157, splat (i32 7)
  %160 = getelementptr i8, ptr %1, i64 992
  store <8 x i32> %158, ptr %151, align 4, !tbaa !6
  store <8 x i32> %159, ptr %160, align 4, !tbaa !6
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z9example24ss(i16 noundef %0, i16 noundef %1) local_unnamed_addr #0 {
  %3 = insertelement <4 x i16> poison, i16 %0, i64 0
  %4 = shufflevector <4 x i16> %3, <4 x i16> poison, <4 x i32> zeroinitializer
  %5 = insertelement <4 x i16> poison, i16 %1, i64 0
  %6 = shufflevector <4 x i16> %5, <4 x i16> poison, <4 x i32> zeroinitializer
  br label %7

7:                                                ; preds = %7, %2
  %8 = phi i64 [ 0, %2 ], [ %25, %7 ]
  %9 = getelementptr inbounds nuw float, ptr @fa, i64 %8
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %11 = load <4 x float>, ptr %9, align 4, !tbaa !77
  %12 = load <4 x float>, ptr %10, align 4, !tbaa !77
  %13 = getelementptr inbounds nuw float, ptr @fb, i64 %8
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %15 = load <4 x float>, ptr %13, align 4, !tbaa !77
  %16 = load <4 x float>, ptr %14, align 4, !tbaa !77
  %17 = fcmp olt <4 x float> %11, %15
  %18 = fcmp olt <4 x float> %12, %16
  %19 = select <4 x i1> %17, <4 x i16> %4, <4 x i16> %6
  %20 = select <4 x i1> %18, <4 x i16> %4, <4 x i16> %6
  %21 = sext <4 x i16> %19 to <4 x i32>
  %22 = sext <4 x i16> %20 to <4 x i32>
  %23 = getelementptr inbounds nuw i32, ptr @ic, i64 %8
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  store <4 x i32> %21, ptr %23, align 16, !tbaa !6
  store <4 x i32> %22, ptr %24, align 16, !tbaa !6
  %25 = add nuw i64 %8, 8
  %26 = icmp eq i64 %25, 1024
  br i1 %26, label %27, label %7, !llvm.loop !79

27:                                               ; preds = %7
  ret void
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @_Z9example25v() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %29, %1 ]
  %3 = getelementptr inbounds nuw float, ptr @da, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x float>, ptr %3, align 4, !tbaa !77
  %6 = load <4 x float>, ptr %4, align 4, !tbaa !77
  %7 = getelementptr inbounds nuw float, ptr @db, i64 %2
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %9 = load <4 x float>, ptr %7, align 4, !tbaa !77
  %10 = load <4 x float>, ptr %8, align 4, !tbaa !77
  %11 = fcmp olt <4 x float> %5, %9
  %12 = fcmp olt <4 x float> %6, %10
  %13 = getelementptr inbounds nuw float, ptr @dc, i64 %2
  %14 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %15 = load <4 x float>, ptr %13, align 4, !tbaa !77
  %16 = load <4 x float>, ptr %14, align 4, !tbaa !77
  %17 = getelementptr inbounds nuw float, ptr @dd, i64 %2
  %18 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %19 = load <4 x float>, ptr %17, align 4, !tbaa !77
  %20 = load <4 x float>, ptr %18, align 4, !tbaa !77
  %21 = fcmp olt <4 x float> %15, %19
  %22 = fcmp olt <4 x float> %16, %20
  %23 = and <4 x i1> %11, %21
  %24 = and <4 x i1> %12, %22
  %25 = zext <4 x i1> %23 to <4 x i32>
  %26 = zext <4 x i1> %24 to <4 x i32>
  %27 = getelementptr inbounds nuw i32, ptr @dj, i64 %2
  %28 = getelementptr inbounds nuw i8, ptr %27, i64 16
  store <4 x i32> %25, ptr %27, align 4, !tbaa !6
  store <4 x i32> %26, ptr %28, align 4, !tbaa !6
  %29 = add nuw i64 %2, 8
  %30 = icmp eq i64 %29, 1024
  br i1 %30, label %31, label %1, !llvm.loop !80

31:                                               ; preds = %1
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: write) uwtable
define dso_local void @_Z11init_memoryPvS_(ptr noundef writeonly captures(address) %0, ptr noundef readnone captures(address) %1) local_unnamed_addr #9 {
  %3 = icmp eq ptr %0, %1
  br i1 %3, label %13, label %4

4:                                                ; preds = %2, %4
  %5 = phi i32 [ %9, %4 ], [ 1, %2 ]
  %6 = phi ptr [ %11, %4 ], [ %0, %2 ]
  %7 = mul i32 %5, 7
  %8 = xor i32 %7, 39
  %9 = add i32 %8, 1
  %10 = trunc i32 %9 to i8
  store i8 %10, ptr %6, align 1, !tbaa !81
  %11 = getelementptr inbounds nuw i8, ptr %6, i64 1
  %12 = icmp eq ptr %11, %1
  br i1 %12, label %13, label %4, !llvm.loop !82

13:                                               ; preds = %4, %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind memory(argmem: write) uwtable
define dso_local void @_Z17init_memory_floatPfS_(ptr noundef writeonly captures(address) %0, ptr noundef readnone captures(address) %1) local_unnamed_addr #9 {
  %3 = icmp eq ptr %0, %1
  br i1 %3, label %12, label %4

4:                                                ; preds = %2, %4
  %5 = phi float [ %9, %4 ], [ 1.000000e+00, %2 ]
  %6 = phi ptr [ %10, %4 ], [ %0, %2 ]
  %7 = fpext float %5 to double
  %8 = fmul double %7, 1.100000e+00
  %9 = fptrunc double %8 to float
  store float %9, ptr %6, align 4, !tbaa !77
  %10 = getelementptr inbounds nuw i8, ptr %6, i64 4
  %11 = icmp eq ptr %10, %1
  br i1 %11, label %12, label %4, !llvm.loop !83

12:                                               ; preds = %4, %2
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local noundef i32 @_Z13digest_memoryPvS_(ptr noundef readonly captures(address) %0, ptr noundef readnone captures(address) %1) local_unnamed_addr #10 {
  %3 = icmp eq ptr %0, %1
  br i1 %3, label %16, label %4

4:                                                ; preds = %2, %4
  %5 = phi i32 [ %13, %4 ], [ 1, %2 ]
  %6 = phi ptr [ %14, %4 ], [ %0, %2 ]
  %7 = mul i32 %5, 3
  %8 = load i8, ptr %6, align 1, !tbaa !81
  %9 = zext i8 %8 to i32
  %10 = xor i32 %7, %9
  %11 = lshr i32 %7, 8
  %12 = shl i32 %10, 8
  %13 = xor i32 %12, %11
  %14 = getelementptr inbounds nuw i8, ptr %6, i64 1
  %15 = icmp eq ptr %14, %1
  br i1 %15, label %16, label %4, !llvm.loop !84

16:                                               ; preds = %4, %2
  %17 = phi i32 [ 1, %2 ], [ %13, %4 ]
  ret i32 %17
}

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #11 personality ptr @__gxx_personality_v0 {
  %3 = alloca i32, align 4
  %4 = alloca %class.Timer, align 8
  %5 = alloca %class.Timer, align 8
  %6 = alloca %class.Timer, align 8
  %7 = alloca %class.Timer, align 8
  %8 = alloca %class.Timer, align 8
  %9 = alloca %class.Timer, align 8
  %10 = alloca %class.Timer, align 8
  %11 = alloca %class.Timer, align 8
  %12 = alloca %class.Timer, align 8
  %13 = alloca %class.Timer, align 8
  %14 = alloca %class.Timer, align 8
  %15 = alloca %class.Timer, align 8
  %16 = alloca %class.Timer, align 8
  %17 = alloca %class.Timer, align 8
  %18 = alloca %class.Timer, align 8
  %19 = alloca %class.Timer, align 8
  %20 = alloca %class.Timer, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #24
  store i32 0, ptr %3, align 4, !tbaa !6
  br label %21

21:                                               ; preds = %21, %2
  %22 = phi i32 [ %27, %21 ], [ 1, %2 ]
  %23 = phi i64 [ %29, %21 ], [ 0, %2 ]
  %24 = getelementptr inbounds nuw i8, ptr @ia, i64 %23
  %25 = mul i32 %22, 7
  %26 = xor i32 %25, 39
  %27 = add i32 %26, 1
  %28 = trunc i32 %27 to i8
  store i8 %28, ptr %24, align 1, !tbaa !81
  %29 = add nuw nsw i64 %23, 1
  %30 = icmp eq i64 %29, 4096
  br i1 %30, label %31, label %21, !llvm.loop !82

31:                                               ; preds = %21, %31
  %32 = phi i32 [ %37, %31 ], [ 1, %21 ]
  %33 = phi i64 [ %39, %31 ], [ 0, %21 ]
  %34 = getelementptr inbounds nuw i8, ptr @ib, i64 %33
  %35 = mul i32 %32, 7
  %36 = xor i32 %35, 39
  %37 = add i32 %36, 1
  %38 = trunc i32 %37 to i8
  store i8 %38, ptr %34, align 1, !tbaa !81
  %39 = add nuw nsw i64 %33, 1
  %40 = icmp eq i64 %39, 4096
  br i1 %40, label %41, label %31, !llvm.loop !82

41:                                               ; preds = %31, %41
  %42 = phi i32 [ %47, %41 ], [ 1, %31 ]
  %43 = phi i64 [ %49, %41 ], [ 0, %31 ]
  %44 = getelementptr inbounds nuw i8, ptr @ic, i64 %43
  %45 = mul i32 %42, 7
  %46 = xor i32 %45, 39
  %47 = add i32 %46, 1
  %48 = trunc i32 %47 to i8
  store i8 %48, ptr %44, align 1, !tbaa !81
  %49 = add nuw nsw i64 %43, 1
  %50 = icmp eq i64 %49, 4096
  br i1 %50, label %51, label %41, !llvm.loop !82

51:                                               ; preds = %41, %51
  %52 = phi i32 [ %57, %51 ], [ 1, %41 ]
  %53 = phi i64 [ %59, %51 ], [ 0, %41 ]
  %54 = getelementptr inbounds nuw i8, ptr @sa, i64 %53
  %55 = mul i32 %52, 7
  %56 = xor i32 %55, 39
  %57 = add i32 %56, 1
  %58 = trunc i32 %57 to i8
  store i8 %58, ptr %54, align 1, !tbaa !81
  %59 = add nuw nsw i64 %53, 1
  %60 = icmp eq i64 %59, 2048
  br i1 %60, label %61, label %51, !llvm.loop !82

61:                                               ; preds = %51, %61
  %62 = phi i32 [ %67, %61 ], [ 1, %51 ]
  %63 = phi i64 [ %69, %61 ], [ 0, %51 ]
  %64 = getelementptr inbounds nuw i8, ptr @sb, i64 %63
  %65 = mul i32 %62, 7
  %66 = xor i32 %65, 39
  %67 = add i32 %66, 1
  %68 = trunc i32 %67 to i8
  store i8 %68, ptr %64, align 1, !tbaa !81
  %69 = add nuw nsw i64 %63, 1
  %70 = icmp eq i64 %69, 2048
  br i1 %70, label %71, label %61, !llvm.loop !82

71:                                               ; preds = %61, %71
  %72 = phi i32 [ %77, %71 ], [ 1, %61 ]
  %73 = phi i64 [ %79, %71 ], [ 0, %61 ]
  %74 = getelementptr inbounds nuw i8, ptr @sc, i64 %73
  %75 = mul i32 %72, 7
  %76 = xor i32 %75, 39
  %77 = add i32 %76, 1
  %78 = trunc i32 %77 to i8
  store i8 %78, ptr %74, align 1, !tbaa !81
  %79 = add nuw nsw i64 %73, 1
  %80 = icmp eq i64 %79, 2048
  br i1 %80, label %81, label %71, !llvm.loop !82

81:                                               ; preds = %71, %81
  %82 = phi i32 [ %87, %81 ], [ 1, %71 ]
  %83 = phi i64 [ %89, %81 ], [ 0, %71 ]
  %84 = getelementptr inbounds nuw i8, ptr @a, i64 %83
  %85 = mul i32 %82, 7
  %86 = xor i32 %85, 39
  %87 = add i32 %86, 1
  %88 = trunc i32 %87 to i8
  store i8 %88, ptr %84, align 1, !tbaa !81
  %89 = add nuw nsw i64 %83, 1
  %90 = icmp eq i64 %89, 8192
  br i1 %90, label %91, label %81, !llvm.loop !82

91:                                               ; preds = %81, %91
  %92 = phi i32 [ %97, %91 ], [ 1, %81 ]
  %93 = phi i64 [ %99, %91 ], [ 0, %81 ]
  %94 = getelementptr inbounds nuw i8, ptr @b, i64 %93
  %95 = mul i32 %92, 7
  %96 = xor i32 %95, 39
  %97 = add i32 %96, 1
  %98 = trunc i32 %97 to i8
  store i8 %98, ptr %94, align 1, !tbaa !81
  %99 = add nuw nsw i64 %93, 1
  %100 = icmp eq i64 %99, 8192
  br i1 %100, label %101, label %91, !llvm.loop !82

101:                                              ; preds = %91, %101
  %102 = phi i32 [ %107, %101 ], [ 1, %91 ]
  %103 = phi i64 [ %109, %101 ], [ 0, %91 ]
  %104 = getelementptr inbounds nuw i8, ptr @c, i64 %103
  %105 = mul i32 %102, 7
  %106 = xor i32 %105, 39
  %107 = add i32 %106, 1
  %108 = trunc i32 %107 to i8
  store i8 %108, ptr %104, align 1, !tbaa !81
  %109 = add nuw nsw i64 %103, 1
  %110 = icmp eq i64 %109, 8192
  br i1 %110, label %111, label %101, !llvm.loop !82

111:                                              ; preds = %101, %111
  %112 = phi i32 [ %117, %111 ], [ 1, %101 ]
  %113 = phi i64 [ %119, %111 ], [ 0, %101 ]
  %114 = getelementptr inbounds nuw i8, ptr @ua, i64 %113
  %115 = mul i32 %112, 7
  %116 = xor i32 %115, 39
  %117 = add i32 %116, 1
  %118 = trunc i32 %117 to i8
  store i8 %118, ptr %114, align 1, !tbaa !81
  %119 = add nuw nsw i64 %113, 1
  %120 = icmp eq i64 %119, 4096
  br i1 %120, label %121, label %111, !llvm.loop !82

121:                                              ; preds = %111, %121
  %122 = phi i32 [ %127, %121 ], [ 1, %111 ]
  %123 = phi i64 [ %129, %121 ], [ 0, %111 ]
  %124 = getelementptr inbounds nuw i8, ptr @ub, i64 %123
  %125 = mul i32 %122, 7
  %126 = xor i32 %125, 39
  %127 = add i32 %126, 1
  %128 = trunc i32 %127 to i8
  store i8 %128, ptr %124, align 1, !tbaa !81
  %129 = add nuw nsw i64 %123, 1
  %130 = icmp eq i64 %129, 4096
  br i1 %130, label %131, label %121, !llvm.loop !82

131:                                              ; preds = %121, %131
  %132 = phi i32 [ %137, %131 ], [ 1, %121 ]
  %133 = phi i64 [ %139, %131 ], [ 0, %121 ]
  %134 = getelementptr inbounds nuw i8, ptr @uc, i64 %133
  %135 = mul i32 %132, 7
  %136 = xor i32 %135, 39
  %137 = add i32 %136, 1
  %138 = trunc i32 %137 to i8
  store i8 %138, ptr %134, align 1, !tbaa !81
  %139 = add nuw nsw i64 %133, 1
  %140 = icmp eq i64 %139, 4096
  br i1 %140, label %141, label %131, !llvm.loop !82

141:                                              ; preds = %131, %141
  %142 = phi i32 [ %147, %141 ], [ 1, %131 ]
  %143 = phi i64 [ %149, %141 ], [ 0, %131 ]
  %144 = getelementptr inbounds nuw i8, ptr @G, i64 %143
  %145 = mul i32 %142, 7
  %146 = xor i32 %145, 39
  %147 = add i32 %146, 1
  %148 = trunc i32 %147 to i8
  store i8 %148, ptr %144, align 1, !tbaa !81
  %149 = add nuw nsw i64 %143, 1
  %150 = icmp eq i64 %149, 4096
  br i1 %150, label %151, label %141, !llvm.loop !82

151:                                              ; preds = %141, %151
  %152 = phi float [ %157, %151 ], [ 1.000000e+00, %141 ]
  %153 = phi i64 [ %158, %151 ], [ 0, %141 ]
  %154 = getelementptr inbounds nuw i8, ptr @fa, i64 %153
  %155 = fpext float %152 to double
  %156 = fmul double %155, 1.100000e+00
  %157 = fptrunc double %156 to float
  store float %157, ptr %154, align 4, !tbaa !77
  %158 = add nuw nsw i64 %153, 4
  %159 = icmp eq i64 %158, 4096
  br i1 %159, label %160, label %151, !llvm.loop !83

160:                                              ; preds = %151, %160
  %161 = phi float [ %166, %160 ], [ 1.000000e+00, %151 ]
  %162 = phi i64 [ %167, %160 ], [ 0, %151 ]
  %163 = getelementptr inbounds nuw i8, ptr @fb, i64 %162
  %164 = fpext float %161 to double
  %165 = fmul double %164, 1.100000e+00
  %166 = fptrunc double %165 to float
  store float %166, ptr %163, align 4, !tbaa !77
  %167 = add nuw nsw i64 %162, 4
  %168 = icmp eq i64 %167, 4096
  br i1 %168, label %169, label %160, !llvm.loop !83

169:                                              ; preds = %160, %169
  %170 = phi float [ %175, %169 ], [ 1.000000e+00, %160 ]
  %171 = phi i64 [ %176, %169 ], [ 0, %160 ]
  %172 = getelementptr inbounds nuw i8, ptr @da, i64 %171
  %173 = fpext float %170 to double
  %174 = fmul double %173, 1.100000e+00
  %175 = fptrunc double %174 to float
  store float %175, ptr %172, align 4, !tbaa !77
  %176 = add nuw nsw i64 %171, 4
  %177 = icmp eq i64 %176, 4096
  br i1 %177, label %178, label %169, !llvm.loop !83

178:                                              ; preds = %169, %178
  %179 = phi float [ %184, %178 ], [ 1.000000e+00, %169 ]
  %180 = phi i64 [ %185, %178 ], [ 0, %169 ]
  %181 = getelementptr inbounds nuw i8, ptr @db, i64 %180
  %182 = fpext float %179 to double
  %183 = fmul double %182, 1.100000e+00
  %184 = fptrunc double %183 to float
  store float %184, ptr %181, align 4, !tbaa !77
  %185 = add nuw nsw i64 %180, 4
  %186 = icmp eq i64 %185, 4096
  br i1 %186, label %187, label %178, !llvm.loop !83

187:                                              ; preds = %178, %187
  %188 = phi float [ %193, %187 ], [ 1.000000e+00, %178 ]
  %189 = phi i64 [ %194, %187 ], [ 0, %178 ]
  %190 = getelementptr inbounds nuw i8, ptr @dc, i64 %189
  %191 = fpext float %188 to double
  %192 = fmul double %191, 1.100000e+00
  %193 = fptrunc double %192 to float
  store float %193, ptr %190, align 4, !tbaa !77
  %194 = add nuw nsw i64 %189, 4
  %195 = icmp eq i64 %194, 4096
  br i1 %195, label %196, label %187, !llvm.loop !83

196:                                              ; preds = %187, %196
  %197 = phi float [ %202, %196 ], [ 1.000000e+00, %187 ]
  %198 = phi i64 [ %203, %196 ], [ 0, %187 ]
  %199 = getelementptr inbounds nuw i8, ptr @dd, i64 %198
  %200 = fpext float %197 to double
  %201 = fmul double %200, 1.100000e+00
  %202 = fptrunc double %201 to float
  store float %202, ptr %199, align 4, !tbaa !77
  %203 = add nuw nsw i64 %198, 4
  %204 = icmp eq i64 %203, 4096
  br i1 %204, label %205, label %196, !llvm.loop !83

205:                                              ; preds = %196
  %206 = icmp sgt i32 %0, 1
  tail call void @_Z8example1v()
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #24
  %207 = zext i1 %206 to i8
  store ptr @.str, ptr %4, align 8, !tbaa !85
  %208 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i8 %207, ptr %208, align 8, !tbaa !91
  %209 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %210 = call i32 @gettimeofday(ptr noundef nonnull %209, ptr noundef null) #24
  br label %226

211:                                              ; preds = %226, %211
  %212 = phi i32 [ %221, %211 ], [ 1, %226 ]
  %213 = phi i64 [ %222, %211 ], [ 0, %226 ]
  %214 = getelementptr inbounds nuw i8, ptr @a, i64 %213
  %215 = mul i32 %212, 3
  %216 = load i8, ptr %214, align 1, !tbaa !81
  %217 = zext i8 %216 to i32
  %218 = xor i32 %215, %217
  %219 = lshr i32 %215, 8
  %220 = shl i32 %218, 8
  %221 = xor i32 %220, %219
  %222 = add nuw nsw i64 %213, 1
  %223 = icmp eq i64 %222, 1024
  br i1 %223, label %224, label %211, !llvm.loop !84

224:                                              ; preds = %211
  %225 = invoke noalias noundef nonnull dereferenceable(4) ptr @_Znwm(i64 noundef 4) #25
          to label %230 unwind label %255

226:                                              ; preds = %205, %226
  %227 = phi i32 [ 0, %205 ], [ %228, %226 ]
  tail call void @_Z8example1v()
  %228 = add nuw nsw i32 %227, 1
  %229 = icmp eq i32 %228, 2621440
  br i1 %229, label %211, label %226, !llvm.loop !92

230:                                              ; preds = %224
  store i32 %221, ptr %225, align 4, !tbaa !6
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %4) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #24
  call void @_Z9example2aii(i32 noundef 1024, i32 noundef 2)
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #24
  store ptr @.str.1, ptr %5, align 8, !tbaa !85
  %231 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store i8 %207, ptr %231, align 8, !tbaa !91
  %232 = getelementptr inbounds nuw i8, ptr %5, i64 16
  %233 = call i32 @gettimeofday(ptr noundef nonnull %232, ptr noundef null) #24
  call void @_Z9example2aii(i32 noundef 1024, i32 noundef 2)
  br label %234

234:                                              ; preds = %230, %234
  %235 = phi i32 [ %244, %234 ], [ 1, %230 ]
  %236 = phi i64 [ %245, %234 ], [ 0, %230 ]
  %237 = getelementptr inbounds nuw i8, ptr @b, i64 %236
  %238 = mul i32 %235, 3
  %239 = load i8, ptr %237, align 1, !tbaa !81
  %240 = zext i8 %239 to i32
  %241 = xor i32 %238, %240
  %242 = lshr i32 %238, 8
  %243 = shl i32 %241, 8
  %244 = xor i32 %243, %242
  %245 = add nuw nsw i64 %236, 1
  %246 = icmp eq i64 %245, 4096
  br i1 %246, label %247, label %234, !llvm.loop !84

247:                                              ; preds = %234
  %248 = invoke noalias noundef nonnull dereferenceable(8) ptr @_Znwm(i64 noundef 8) #25
          to label %249 unwind label %278

249:                                              ; preds = %247
  %250 = getelementptr inbounds nuw i8, ptr %248, i64 4
  store i32 %244, ptr %250, align 4, !tbaa !6
  %251 = load i32, ptr %225, align 4
  store i32 %251, ptr %248, align 4
  call void @_ZdlPvm(ptr noundef nonnull %225, i64 noundef 4) #26
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %5) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #24
  call void @_Z9example2bii(i32 noundef 1024, i32 poison)
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #24
  store ptr @.str.2, ptr %6, align 8, !tbaa !85
  %252 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i8 %207, ptr %252, align 8, !tbaa !91
  %253 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %254 = call i32 @gettimeofday(ptr noundef nonnull %253, ptr noundef null) #24
  br label %281

255:                                              ; preds = %224
  %256 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %4) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #24
  br label %875

257:                                              ; preds = %281, %257
  %258 = phi i32 [ %267, %257 ], [ 1, %281 ]
  %259 = phi i64 [ %268, %257 ], [ 0, %281 ]
  %260 = getelementptr inbounds nuw i8, ptr @a, i64 %259
  %261 = mul i32 %258, 3
  %262 = load i8, ptr %260, align 1, !tbaa !81
  %263 = zext i8 %262 to i32
  %264 = xor i32 %261, %263
  %265 = lshr i32 %261, 8
  %266 = shl i32 %264, 8
  %267 = xor i32 %266, %265
  %268 = add nuw nsw i64 %259, 1
  %269 = icmp eq i64 %268, 4096
  br i1 %269, label %270, label %257, !llvm.loop !84

270:                                              ; preds = %257
  %271 = invoke noalias noundef nonnull dereferenceable(16) ptr @_Znwm(i64 noundef 16) #25
          to label %272 unwind label %298

272:                                              ; preds = %270
  %273 = getelementptr inbounds nuw i8, ptr %271, i64 8
  store i32 %267, ptr %273, align 4, !tbaa !6
  %274 = load i64, ptr %248, align 4
  store i64 %274, ptr %271, align 4
  call void @_ZdlPvm(ptr noundef nonnull %248, i64 noundef 8) #26
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %6) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #24
  call void @_Z8example3iPiS_(i32 noundef 1024, ptr noundef nonnull @ia, ptr noundef nonnull @ib)
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #24
  store ptr @.str.3, ptr %7, align 8, !tbaa !85
  %275 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i8 %207, ptr %275, align 8, !tbaa !91
  %276 = getelementptr inbounds nuw i8, ptr %7, i64 16
  %277 = call i32 @gettimeofday(ptr noundef nonnull %276, ptr noundef null) #24
  br label %301

278:                                              ; preds = %247
  %279 = landingpad { ptr, i32 }
          cleanup
  %280 = getelementptr inbounds nuw i8, ptr %225, i64 4
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %5) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #24
  br label %868

281:                                              ; preds = %249, %281
  %282 = phi i32 [ 0, %249 ], [ %283, %281 ]
  call void @_Z9example2bii(i32 noundef 1024, i32 poison)
  %283 = add nuw nsw i32 %282, 1
  %284 = icmp eq i32 %283, 524288
  br i1 %284, label %257, label %281, !llvm.loop !93

285:                                              ; preds = %301, %285
  %286 = phi i32 [ %295, %285 ], [ 1, %301 ]
  %287 = phi i64 [ %296, %285 ], [ 0, %301 ]
  %288 = getelementptr inbounds nuw i8, ptr @ia, i64 %287
  %289 = mul i32 %286, 3
  %290 = load i8, ptr %288, align 1, !tbaa !81
  %291 = zext i8 %290 to i32
  %292 = xor i32 %289, %291
  %293 = lshr i32 %289, 8
  %294 = shl i32 %292, 8
  %295 = xor i32 %294, %293
  %296 = add nuw nsw i64 %287, 1
  %297 = icmp eq i64 %296, 4096
  br i1 %297, label %305, label %285, !llvm.loop !84

298:                                              ; preds = %270
  %299 = landingpad { ptr, i32 }
          cleanup
  %300 = getelementptr inbounds nuw i8, ptr %248, i64 8
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %6) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #24
  br label %868

301:                                              ; preds = %272, %301
  %302 = phi i32 [ 0, %272 ], [ %303, %301 ]
  call void @_Z8example3iPiS_(i32 noundef 1024, ptr noundef nonnull @ia, ptr noundef nonnull @ib)
  %303 = add nuw nsw i32 %302, 1
  %304 = icmp eq i32 %303, 524288
  br i1 %304, label %285, label %301, !llvm.loop !94

305:                                              ; preds = %285
  %306 = getelementptr inbounds nuw i8, ptr %271, i64 16
  %307 = getelementptr inbounds nuw i8, ptr %271, i64 12
  store i32 %295, ptr %307, align 4, !tbaa !6
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %7) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #24
  call void @_Z9example4aiPiS_(i32 noundef 1024, ptr noundef nonnull @ia, ptr noundef nonnull @ib)
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #24
  store ptr @.str.4, ptr %8, align 8, !tbaa !85
  %308 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store i8 %207, ptr %308, align 8, !tbaa !91
  %309 = getelementptr inbounds nuw i8, ptr %8, i64 16
  %310 = call i32 @gettimeofday(ptr noundef nonnull %309, ptr noundef null) #24
  br label %326

311:                                              ; preds = %326, %311
  %312 = phi i32 [ %321, %311 ], [ 1, %326 ]
  %313 = phi i64 [ %322, %311 ], [ 0, %326 ]
  %314 = getelementptr inbounds nuw i8, ptr @ia, i64 %313
  %315 = mul i32 %312, 3
  %316 = load i8, ptr %314, align 1, !tbaa !81
  %317 = zext i8 %316 to i32
  %318 = xor i32 %315, %317
  %319 = lshr i32 %315, 8
  %320 = shl i32 %318, 8
  %321 = xor i32 %320, %319
  %322 = add nuw nsw i64 %313, 1
  %323 = icmp eq i64 %322, 4096
  br i1 %323, label %324, label %311, !llvm.loop !84

324:                                              ; preds = %311
  %325 = invoke noalias noundef nonnull dereferenceable(32) ptr @_Znwm(i64 noundef 32) #25
          to label %330 unwind label %349

326:                                              ; preds = %305, %326
  %327 = phi i32 [ 0, %305 ], [ %328, %326 ]
  call void @_Z9example4aiPiS_(i32 noundef 1024, ptr noundef nonnull @ia, ptr noundef nonnull @ib)
  %328 = add nuw nsw i32 %327, 1
  %329 = icmp eq i32 %328, 524288
  br i1 %329, label %311, label %326, !llvm.loop !95

330:                                              ; preds = %324
  %331 = getelementptr inbounds nuw i8, ptr %325, i64 16
  store i32 %321, ptr %331, align 4, !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(16) %325, ptr noundef nonnull align 4 dereferenceable(16) %271, i64 16, i1 false)
  call void @_ZdlPvm(ptr noundef nonnull %271, i64 noundef 16) #26
  %332 = getelementptr inbounds nuw i8, ptr %325, i64 32
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %8) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #24
  call void @_Z9example4biPiS_(i32 noundef 1014, ptr nonnull poison, ptr nonnull poison)
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #24
  store ptr @.str.5, ptr %9, align 8, !tbaa !85
  %333 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store i8 %207, ptr %333, align 8, !tbaa !91
  %334 = getelementptr inbounds nuw i8, ptr %9, i64 16
  %335 = call i32 @gettimeofday(ptr noundef nonnull %334, ptr noundef null) #24
  br label %351

336:                                              ; preds = %351, %336
  %337 = phi i32 [ %346, %336 ], [ 1, %351 ]
  %338 = phi i64 [ %347, %336 ], [ 0, %351 ]
  %339 = getelementptr inbounds nuw i8, ptr @ia, i64 %338
  %340 = mul i32 %337, 3
  %341 = load i8, ptr %339, align 1, !tbaa !81
  %342 = zext i8 %341 to i32
  %343 = xor i32 %340, %342
  %344 = lshr i32 %340, 8
  %345 = shl i32 %343, 8
  %346 = xor i32 %345, %344
  %347 = add nuw nsw i64 %338, 1
  %348 = icmp eq i64 %347, 4096
  br i1 %348, label %355, label %336, !llvm.loop !84

349:                                              ; preds = %324
  %350 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %8) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #24
  br label %868

351:                                              ; preds = %330, %351
  %352 = phi i32 [ 0, %330 ], [ %353, %351 ]
  call void @_Z9example4biPiS_(i32 noundef 1014, ptr nonnull poison, ptr nonnull poison)
  %353 = add nuw nsw i32 %352, 1
  %354 = icmp eq i32 %353, 524288
  br i1 %354, label %336, label %351, !llvm.loop !96

355:                                              ; preds = %336
  %356 = getelementptr inbounds nuw i8, ptr %325, i64 20
  store i32 %346, ptr %356, align 4, !tbaa !6
  %357 = getelementptr inbounds nuw i8, ptr %325, i64 24
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %9) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #24
  call void @_Z9example4ciPiS_(i32 noundef 1024, ptr nonnull poison, ptr nonnull poison)
  call void @llvm.lifetime.start.p0(ptr nonnull %10) #24
  store ptr @.str.6, ptr %10, align 8, !tbaa !85
  %358 = getelementptr inbounds nuw i8, ptr %10, i64 8
  store i8 %207, ptr %358, align 8, !tbaa !91
  %359 = getelementptr inbounds nuw i8, ptr %10, i64 16
  %360 = call i32 @gettimeofday(ptr noundef nonnull %359, ptr noundef null) #24
  br label %378

361:                                              ; preds = %378, %361
  %362 = phi i32 [ %371, %361 ], [ 1, %378 ]
  %363 = phi i64 [ %372, %361 ], [ 0, %378 ]
  %364 = getelementptr inbounds nuw i8, ptr @ib, i64 %363
  %365 = mul i32 %362, 3
  %366 = load i8, ptr %364, align 1, !tbaa !81
  %367 = zext i8 %366 to i32
  %368 = xor i32 %365, %367
  %369 = lshr i32 %365, 8
  %370 = shl i32 %368, 8
  %371 = xor i32 %370, %369
  %372 = add nuw nsw i64 %363, 1
  %373 = icmp eq i64 %372, 4096
  br i1 %373, label %374, label %361, !llvm.loop !84

374:                                              ; preds = %361
  store i32 %371, ptr %357, align 4, !tbaa !6
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %10) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %10) #24
  call void @_Z8example7i(i32 noundef 4)
  call void @llvm.lifetime.start.p0(ptr nonnull %11) #24
  store ptr @.str.7, ptr %11, align 8, !tbaa !85
  %375 = getelementptr inbounds nuw i8, ptr %11, i64 8
  store i8 %207, ptr %375, align 8, !tbaa !91
  %376 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %377 = call i32 @gettimeofday(ptr noundef nonnull %376, ptr noundef null) #24
  br label %400

378:                                              ; preds = %355, %378
  %379 = phi i32 [ 0, %355 ], [ %380, %378 ]
  call void @_Z9example4ciPiS_(i32 noundef 1024, ptr nonnull poison, ptr nonnull poison)
  %380 = add nuw nsw i32 %379, 1
  %381 = icmp eq i32 %380, 524288
  br i1 %381, label %361, label %378, !llvm.loop !97

382:                                              ; preds = %400, %382
  %383 = phi i32 [ %392, %382 ], [ 1, %400 ]
  %384 = phi i64 [ %393, %382 ], [ 0, %400 ]
  %385 = getelementptr inbounds nuw i8, ptr @a, i64 %384
  %386 = mul i32 %383, 3
  %387 = load i8, ptr %385, align 1, !tbaa !81
  %388 = zext i8 %387 to i32
  %389 = xor i32 %386, %388
  %390 = lshr i32 %386, 8
  %391 = shl i32 %389, 8
  %392 = xor i32 %391, %390
  %393 = add nuw nsw i64 %384, 1
  %394 = icmp eq i64 %393, 4096
  br i1 %394, label %395, label %382, !llvm.loop !84

395:                                              ; preds = %382
  %396 = getelementptr inbounds nuw i8, ptr %325, i64 28
  store i32 %392, ptr %396, align 4, !tbaa !6
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %11) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %11) #24
  call void @_Z8example8i(i32 noundef 8)
  call void @llvm.lifetime.start.p0(ptr nonnull %12) #24
  store ptr @.str.8, ptr %12, align 8, !tbaa !85
  %397 = getelementptr inbounds nuw i8, ptr %12, i64 8
  store i8 %207, ptr %397, align 8, !tbaa !91
  %398 = getelementptr inbounds nuw i8, ptr %12, i64 16
  %399 = call i32 @gettimeofday(ptr noundef nonnull %398, ptr noundef null) #24
  call void @_Z8example8i(i32 noundef 8)
  br label %404

400:                                              ; preds = %374, %400
  %401 = phi i32 [ 0, %374 ], [ %402, %400 ]
  call void @_Z8example7i(i32 noundef 4)
  %402 = add nuw nsw i32 %401, 1
  %403 = icmp eq i32 %402, 1048576
  br i1 %403, label %382, label %400, !llvm.loop !98

404:                                              ; preds = %395, %404
  %405 = phi i32 [ %414, %404 ], [ 1, %395 ]
  %406 = phi i64 [ %415, %404 ], [ 0, %395 ]
  %407 = getelementptr inbounds nuw i8, ptr @G, i64 %406
  %408 = mul i32 %405, 3
  %409 = load i8, ptr %407, align 1, !tbaa !81
  %410 = zext i8 %409 to i32
  %411 = xor i32 %408, %410
  %412 = lshr i32 %408, 8
  %413 = shl i32 %411, 8
  %414 = xor i32 %413, %412
  %415 = add nuw nsw i64 %406, 1
  %416 = icmp eq i64 %415, 4096
  br i1 %416, label %417, label %404, !llvm.loop !84

417:                                              ; preds = %404
  %418 = invoke noalias noundef nonnull dereferenceable(64) ptr @_Znwm(i64 noundef 64) #25
          to label %419 unwind label %432

419:                                              ; preds = %417
  %420 = getelementptr inbounds nuw i8, ptr %418, i64 32
  store i32 %414, ptr %420, align 4, !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(32) %418, ptr noundef nonnull align 4 dereferenceable(32) %325, i64 32, i1 false)
  call void @_ZdlPvm(ptr noundef nonnull %325, i64 noundef 32) #26
  %421 = getelementptr inbounds nuw i8, ptr %418, i64 64
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %12) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #24
  call void @_Z8example9Pj(ptr noundef nonnull %3)
  call void @llvm.lifetime.start.p0(ptr nonnull %13) #24
  store ptr @.str.9, ptr %13, align 8, !tbaa !85
  %422 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store i8 %207, ptr %422, align 8, !tbaa !91
  %423 = getelementptr inbounds nuw i8, ptr %13, i64 16
  %424 = call i32 @gettimeofday(ptr noundef nonnull %423, ptr noundef null) #24
  br label %434

425:                                              ; preds = %434
  %426 = load i32, ptr %3, align 4, !tbaa !6
  %427 = getelementptr inbounds nuw i8, ptr %418, i64 36
  store i32 %426, ptr %427, align 4, !tbaa !6
  %428 = getelementptr inbounds nuw i8, ptr %418, i64 40
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %13) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %13) #24
  call void @_Z10example10aPsS_S_PiS0_S0_(ptr noundef nonnull @sa, ptr noundef nonnull @sb, ptr noundef nonnull @sc, ptr noundef nonnull @ia, ptr noundef nonnull @ib, ptr noundef nonnull @ic)
  call void @llvm.lifetime.start.p0(ptr nonnull %14) #24
  store ptr @.str.10, ptr %14, align 8, !tbaa !85
  %429 = getelementptr inbounds nuw i8, ptr %14, i64 8
  store i8 %207, ptr %429, align 8, !tbaa !91
  %430 = getelementptr inbounds nuw i8, ptr %14, i64 16
  %431 = call i32 @gettimeofday(ptr noundef nonnull %430, ptr noundef null) #24
  br label %473

432:                                              ; preds = %417
  %433 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %12) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %12) #24
  br label %868

434:                                              ; preds = %419, %434
  %435 = phi i32 [ 0, %419 ], [ %436, %434 ]
  call void @_Z8example9Pj(ptr noundef nonnull %3)
  %436 = add nuw nsw i32 %435, 1
  %437 = icmp eq i32 %436, 524288
  br i1 %437, label %425, label %434, !llvm.loop !99

438:                                              ; preds = %473, %438
  %439 = phi i32 [ %448, %438 ], [ 1, %473 ]
  %440 = phi i64 [ %449, %438 ], [ 0, %473 ]
  %441 = getelementptr inbounds nuw i8, ptr @ia, i64 %440
  %442 = mul i32 %439, 3
  %443 = load i8, ptr %441, align 1, !tbaa !81
  %444 = zext i8 %443 to i32
  %445 = xor i32 %442, %444
  %446 = lshr i32 %442, 8
  %447 = shl i32 %445, 8
  %448 = xor i32 %447, %446
  %449 = add nuw nsw i64 %440, 1
  %450 = icmp eq i64 %449, 4096
  br i1 %450, label %451, label %438, !llvm.loop !84

451:                                              ; preds = %438, %451
  %452 = phi i32 [ %461, %451 ], [ 1, %438 ]
  %453 = phi i64 [ %462, %451 ], [ 0, %438 ]
  %454 = getelementptr inbounds nuw i8, ptr @sa, i64 %453
  %455 = mul i32 %452, 3
  %456 = load i8, ptr %454, align 1, !tbaa !81
  %457 = zext i8 %456 to i32
  %458 = xor i32 %455, %457
  %459 = lshr i32 %455, 8
  %460 = shl i32 %458, 8
  %461 = xor i32 %460, %459
  %462 = add nuw nsw i64 %453, 1
  %463 = icmp eq i64 %462, 2048
  br i1 %463, label %464, label %451, !llvm.loop !84

464:                                              ; preds = %451
  %465 = add i32 %461, %448
  %466 = icmp eq ptr %428, %421
  br i1 %466, label %468, label %467

467:                                              ; preds = %464
  store i32 %465, ptr %428, align 4, !tbaa !6
  br label %477

468:                                              ; preds = %464
  %469 = invoke noalias noundef nonnull dereferenceable(128) ptr @_Znwm(i64 noundef 128) #25
          to label %470 unwind label %526

470:                                              ; preds = %468
  %471 = getelementptr inbounds nuw i8, ptr %469, i64 64
  store i32 %465, ptr %471, align 4, !tbaa !6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 4 dereferenceable(64) %469, ptr noundef nonnull align 4 dereferenceable(64) %418, i64 64, i1 false)
  call void @_ZdlPvm(ptr noundef nonnull %418, i64 noundef 64) #26
  %472 = getelementptr inbounds nuw i8, ptr %469, i64 128
  br label %477

473:                                              ; preds = %425, %473
  %474 = phi i32 [ 0, %425 ], [ %475, %473 ]
  call void @_Z10example10aPsS_S_PiS0_S0_(ptr noundef nonnull @sa, ptr noundef nonnull @sb, ptr noundef nonnull @sc, ptr noundef nonnull @ia, ptr noundef nonnull @ib, ptr noundef nonnull @ic)
  %475 = add nuw nsw i32 %474, 1
  %476 = icmp eq i32 %475, 524288
  br i1 %476, label %438, label %473, !llvm.loop !100

477:                                              ; preds = %470, %467
  %478 = phi ptr [ %472, %470 ], [ %421, %467 ]
  %479 = phi ptr [ %471, %470 ], [ %428, %467 ]
  %480 = phi ptr [ %469, %470 ], [ %418, %467 ]
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %14) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %14) #24
  call void @_Z10example10bPsS_S_PiS0_S0_(ptr nonnull poison, ptr noundef nonnull @sb, ptr nonnull poison, ptr noundef nonnull @ia, ptr nonnull poison, ptr nonnull poison)
  call void @llvm.lifetime.start.p0(ptr nonnull %15) #24
  store ptr @.str.11, ptr %15, align 8, !tbaa !85
  %481 = getelementptr inbounds nuw i8, ptr %15, i64 8
  store i8 %207, ptr %481, align 8, !tbaa !91
  %482 = getelementptr inbounds nuw i8, ptr %15, i64 16
  %483 = call i32 @gettimeofday(ptr noundef nonnull %482, ptr noundef null) #24
  br label %528

484:                                              ; preds = %528, %484
  %485 = phi i32 [ %494, %484 ], [ 1, %528 ]
  %486 = phi i64 [ %495, %484 ], [ 0, %528 ]
  %487 = getelementptr inbounds nuw i8, ptr @ia, i64 %486
  %488 = mul i32 %485, 3
  %489 = load i8, ptr %487, align 1, !tbaa !81
  %490 = zext i8 %489 to i32
  %491 = xor i32 %488, %490
  %492 = lshr i32 %488, 8
  %493 = shl i32 %491, 8
  %494 = xor i32 %493, %492
  %495 = add nuw nsw i64 %486, 1
  %496 = icmp eq i64 %495, 4096
  br i1 %496, label %497, label %484, !llvm.loop !84

497:                                              ; preds = %484
  %498 = getelementptr inbounds nuw i8, ptr %479, i64 4
  %499 = icmp eq ptr %498, %478
  br i1 %499, label %502, label %500

500:                                              ; preds = %497
  store i32 %494, ptr %498, align 4, !tbaa !6
  %501 = getelementptr inbounds nuw i8, ptr %479, i64 8
  br label %532

502:                                              ; preds = %497
  %503 = ptrtoint ptr %478 to i64
  %504 = ptrtoint ptr %480 to i64
  %505 = sub i64 %503, %504
  %506 = icmp eq i64 %505, 9223372036854775804
  br i1 %506, label %507, label %509

507:                                              ; preds = %502
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.23) #27
          to label %508 unwind label %578

508:                                              ; preds = %507
  unreachable

509:                                              ; preds = %502
  %510 = ashr exact i64 %505, 2
  %511 = call i64 @llvm.umax.i64(i64 %510, i64 1)
  %512 = add nsw i64 %511, %510
  %513 = icmp ult i64 %512, %510
  %514 = call i64 @llvm.umin.i64(i64 %512, i64 2305843009213693951)
  %515 = select i1 %513, i64 2305843009213693951, i64 %514
  %516 = icmp ne i64 %515, 0
  call void @llvm.assume(i1 %516)
  %517 = shl nuw nsw i64 %515, 2
  %518 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %517) #25
          to label %519 unwind label %578

519:                                              ; preds = %509
  %520 = getelementptr inbounds i8, ptr %518, i64 %505
  store i32 %494, ptr %520, align 4, !tbaa !6
  %521 = icmp sgt i64 %505, 0
  br i1 %521, label %522, label %523

522:                                              ; preds = %519
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 %518, ptr nonnull align 4 %480, i64 %505, i1 false)
  br label %523

523:                                              ; preds = %522, %519
  call void @_ZdlPvm(ptr noundef nonnull %480, i64 noundef %505) #26
  %524 = getelementptr inbounds nuw i8, ptr %520, i64 4
  %525 = getelementptr inbounds nuw i32, ptr %518, i64 %515
  br label %532

526:                                              ; preds = %468
  %527 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %14) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %14) #24
  br label %868

528:                                              ; preds = %477, %528
  %529 = phi i32 [ 0, %477 ], [ %530, %528 ]
  call void @_Z10example10bPsS_S_PiS0_S0_(ptr nonnull poison, ptr noundef nonnull @sb, ptr nonnull poison, ptr noundef nonnull @ia, ptr nonnull poison, ptr nonnull poison)
  %530 = add nuw nsw i32 %529, 1
  %531 = icmp eq i32 %530, 1048576
  br i1 %531, label %484, label %528, !llvm.loop !101

532:                                              ; preds = %523, %500
  %533 = phi ptr [ %525, %523 ], [ %478, %500 ]
  %534 = phi ptr [ %524, %523 ], [ %501, %500 ]
  %535 = phi ptr [ %518, %523 ], [ %480, %500 ]
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %15) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %15) #24
  call void @_Z9example11v()
  call void @llvm.lifetime.start.p0(ptr nonnull %16) #24
  store ptr @.str.12, ptr %16, align 8, !tbaa !85
  %536 = getelementptr inbounds nuw i8, ptr %16, i64 8
  store i8 %207, ptr %536, align 8, !tbaa !91
  %537 = getelementptr inbounds nuw i8, ptr %16, i64 16
  %538 = call i32 @gettimeofday(ptr noundef nonnull %537, ptr noundef null) #24
  br label %580

539:                                              ; preds = %580, %539
  %540 = phi i32 [ %549, %539 ], [ 1, %580 ]
  %541 = phi i64 [ %550, %539 ], [ 0, %580 ]
  %542 = getelementptr inbounds nuw i8, ptr @d, i64 %541
  %543 = mul i32 %540, 3
  %544 = load i8, ptr %542, align 1, !tbaa !81
  %545 = zext i8 %544 to i32
  %546 = xor i32 %543, %545
  %547 = lshr i32 %543, 8
  %548 = shl i32 %546, 8
  %549 = xor i32 %548, %547
  %550 = add nuw nsw i64 %541, 1
  %551 = icmp eq i64 %550, 4096
  br i1 %551, label %552, label %539, !llvm.loop !84

552:                                              ; preds = %539
  %553 = icmp eq ptr %534, %533
  br i1 %553, label %555, label %554

554:                                              ; preds = %552
  store i32 %549, ptr %534, align 4, !tbaa !6
  br label %584

555:                                              ; preds = %552
  %556 = ptrtoint ptr %533 to i64
  %557 = ptrtoint ptr %535 to i64
  %558 = sub i64 %556, %557
  %559 = icmp eq i64 %558, 9223372036854775804
  br i1 %559, label %560, label %562

560:                                              ; preds = %555
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.23) #27
          to label %561 unwind label %633

561:                                              ; preds = %560
  unreachable

562:                                              ; preds = %555
  %563 = ashr exact i64 %558, 2
  %564 = call i64 @llvm.umax.i64(i64 %563, i64 1)
  %565 = add nsw i64 %564, %563
  %566 = icmp ult i64 %565, %563
  %567 = call i64 @llvm.umin.i64(i64 %565, i64 2305843009213693951)
  %568 = select i1 %566, i64 2305843009213693951, i64 %567
  %569 = icmp ne i64 %568, 0
  call void @llvm.assume(i1 %569)
  %570 = shl nuw nsw i64 %568, 2
  %571 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %570) #25
          to label %572 unwind label %633

572:                                              ; preds = %562
  %573 = getelementptr inbounds i8, ptr %571, i64 %558
  store i32 %549, ptr %573, align 4, !tbaa !6
  %574 = icmp sgt i64 %558, 0
  br i1 %574, label %575, label %576

575:                                              ; preds = %572
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 %571, ptr nonnull align 4 %535, i64 %558, i1 false)
  br label %576

576:                                              ; preds = %575, %572
  call void @_ZdlPvm(ptr noundef nonnull %535, i64 noundef %558) #26
  %577 = getelementptr inbounds nuw i32, ptr %571, i64 %568
  br label %584

578:                                              ; preds = %509, %507
  %579 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %15) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %15) #24
  br label %868

580:                                              ; preds = %532, %580
  %581 = phi i32 [ 0, %532 ], [ %582, %580 ]
  call void @_Z9example11v()
  %582 = add nuw nsw i32 %581, 1
  %583 = icmp eq i32 %582, 524288
  br i1 %583, label %539, label %580, !llvm.loop !102

584:                                              ; preds = %576, %554
  %585 = phi ptr [ %577, %576 ], [ %533, %554 ]
  %586 = phi ptr [ %573, %576 ], [ %534, %554 ]
  %587 = phi ptr [ %571, %576 ], [ %535, %554 ]
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %16) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %16) #24
  call void @_Z9example12v()
  call void @llvm.lifetime.start.p0(ptr nonnull %17) #24
  store ptr @.str.13, ptr %17, align 8, !tbaa !85
  %588 = getelementptr inbounds nuw i8, ptr %17, i64 8
  store i8 %207, ptr %588, align 8, !tbaa !91
  %589 = getelementptr inbounds nuw i8, ptr %17, i64 16
  %590 = call i32 @gettimeofday(ptr noundef nonnull %589, ptr noundef null) #24
  call void @_Z9example12v()
  br label %591

591:                                              ; preds = %584, %591
  %592 = phi i32 [ %601, %591 ], [ 1, %584 ]
  %593 = phi i64 [ %602, %591 ], [ 0, %584 ]
  %594 = getelementptr inbounds nuw i8, ptr @a, i64 %593
  %595 = mul i32 %592, 3
  %596 = load i8, ptr %594, align 1, !tbaa !81
  %597 = zext i8 %596 to i32
  %598 = xor i32 %595, %597
  %599 = lshr i32 %595, 8
  %600 = shl i32 %598, 8
  %601 = xor i32 %600, %599
  %602 = add nuw nsw i64 %593, 1
  %603 = icmp eq i64 %602, 4096
  br i1 %603, label %604, label %591, !llvm.loop !84

604:                                              ; preds = %591
  %605 = getelementptr inbounds nuw i8, ptr %586, i64 4
  %606 = icmp eq ptr %605, %585
  br i1 %606, label %609, label %607

607:                                              ; preds = %604
  store i32 %601, ptr %605, align 4, !tbaa !6
  %608 = getelementptr inbounds nuw i8, ptr %586, i64 8
  br label %635

609:                                              ; preds = %604
  %610 = ptrtoint ptr %585 to i64
  %611 = ptrtoint ptr %587 to i64
  %612 = sub i64 %610, %611
  %613 = icmp eq i64 %612, 9223372036854775804
  br i1 %613, label %614, label %616

614:                                              ; preds = %609
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.23) #27
          to label %615 unwind label %681

615:                                              ; preds = %614
  unreachable

616:                                              ; preds = %609
  %617 = ashr exact i64 %612, 2
  %618 = call i64 @llvm.umax.i64(i64 %617, i64 1)
  %619 = add nsw i64 %618, %617
  %620 = icmp ult i64 %619, %617
  %621 = call i64 @llvm.umin.i64(i64 %619, i64 2305843009213693951)
  %622 = select i1 %620, i64 2305843009213693951, i64 %621
  %623 = icmp ne i64 %622, 0
  call void @llvm.assume(i1 %623)
  %624 = shl nuw nsw i64 %622, 2
  %625 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %624) #25
          to label %626 unwind label %681

626:                                              ; preds = %616
  %627 = getelementptr inbounds i8, ptr %625, i64 %612
  store i32 %601, ptr %627, align 4, !tbaa !6
  %628 = icmp sgt i64 %612, 0
  br i1 %628, label %629, label %630

629:                                              ; preds = %626
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 %625, ptr nonnull align 4 %587, i64 %612, i1 false)
  br label %630

630:                                              ; preds = %629, %626
  call void @_ZdlPvm(ptr noundef nonnull %587, i64 noundef %612) #26
  %631 = getelementptr inbounds nuw i8, ptr %627, i64 4
  %632 = getelementptr inbounds nuw i32, ptr %625, i64 %622
  br label %635

633:                                              ; preds = %562, %560
  %634 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %16) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %16) #24
  br label %868

635:                                              ; preds = %630, %607
  %636 = phi ptr [ %632, %630 ], [ %585, %607 ]
  %637 = phi ptr [ %631, %630 ], [ %608, %607 ]
  %638 = phi ptr [ %625, %630 ], [ %587, %607 ]
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %17) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %17) #24
  call void @_Z9example23PtPj(ptr noundef nonnull @usa, ptr noundef nonnull @ua)
  call void @llvm.lifetime.start.p0(ptr nonnull %18) #24
  store ptr @.str.14, ptr %18, align 8, !tbaa !85
  %639 = getelementptr inbounds nuw i8, ptr %18, i64 8
  store i8 %207, ptr %639, align 8, !tbaa !91
  %640 = getelementptr inbounds nuw i8, ptr %18, i64 16
  %641 = call i32 @gettimeofday(ptr noundef nonnull %640, ptr noundef null) #24
  br label %683

642:                                              ; preds = %683, %642
  %643 = phi i32 [ %652, %642 ], [ 1, %683 ]
  %644 = phi i64 [ %653, %642 ], [ 0, %683 ]
  %645 = getelementptr inbounds nuw i8, ptr @usa, i64 %644
  %646 = mul i32 %643, 3
  %647 = load i8, ptr %645, align 1, !tbaa !81
  %648 = zext i8 %647 to i32
  %649 = xor i32 %646, %648
  %650 = lshr i32 %646, 8
  %651 = shl i32 %649, 8
  %652 = xor i32 %651, %650
  %653 = add nuw nsw i64 %644, 1
  %654 = icmp eq i64 %653, 512
  br i1 %654, label %655, label %642, !llvm.loop !84

655:                                              ; preds = %642
  %656 = icmp eq ptr %637, %636
  br i1 %656, label %658, label %657

657:                                              ; preds = %655
  store i32 %652, ptr %637, align 4, !tbaa !6
  br label %687

658:                                              ; preds = %655
  %659 = ptrtoint ptr %636 to i64
  %660 = ptrtoint ptr %638 to i64
  %661 = sub i64 %659, %660
  %662 = icmp eq i64 %661, 9223372036854775804
  br i1 %662, label %663, label %665

663:                                              ; preds = %658
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.23) #27
          to label %664 unwind label %723

664:                                              ; preds = %663
  unreachable

665:                                              ; preds = %658
  %666 = ashr exact i64 %661, 2
  %667 = call i64 @llvm.umax.i64(i64 %666, i64 1)
  %668 = add nsw i64 %667, %666
  %669 = icmp ult i64 %668, %666
  %670 = call i64 @llvm.umin.i64(i64 %668, i64 2305843009213693951)
  %671 = select i1 %669, i64 2305843009213693951, i64 %670
  %672 = icmp ne i64 %671, 0
  call void @llvm.assume(i1 %672)
  %673 = shl nuw nsw i64 %671, 2
  %674 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %673) #25
          to label %675 unwind label %723

675:                                              ; preds = %665
  %676 = getelementptr inbounds i8, ptr %674, i64 %661
  store i32 %652, ptr %676, align 4, !tbaa !6
  %677 = icmp sgt i64 %661, 0
  br i1 %677, label %678, label %679

678:                                              ; preds = %675
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 %674, ptr nonnull align 4 %638, i64 %661, i1 false)
  br label %679

679:                                              ; preds = %678, %675
  call void @_ZdlPvm(ptr noundef nonnull %638, i64 noundef %661) #26
  %680 = getelementptr inbounds nuw i32, ptr %674, i64 %671
  br label %687

681:                                              ; preds = %616, %614
  %682 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %17) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %17) #24
  br label %868

683:                                              ; preds = %635, %683
  %684 = phi i32 [ 0, %635 ], [ %685, %683 ]
  call void @_Z9example23PtPj(ptr noundef nonnull @usa, ptr noundef nonnull @ua)
  %685 = add nuw nsw i32 %684, 1
  %686 = icmp eq i32 %685, 2097152
  br i1 %686, label %642, label %683, !llvm.loop !103

687:                                              ; preds = %679, %657
  %688 = phi ptr [ %680, %679 ], [ %636, %657 ]
  %689 = phi ptr [ %676, %679 ], [ %637, %657 ]
  %690 = phi ptr [ %674, %679 ], [ %638, %657 ]
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %18) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %18) #24
  call void @_Z9example24ss(i16 noundef 2, i16 noundef 4)
  call void @llvm.lifetime.start.p0(ptr nonnull %19) #24
  store ptr @.str.15, ptr %19, align 8, !tbaa !85
  %691 = getelementptr inbounds nuw i8, ptr %19, i64 8
  store i8 %207, ptr %691, align 8, !tbaa !91
  %692 = getelementptr inbounds nuw i8, ptr %19, i64 16
  %693 = call i32 @gettimeofday(ptr noundef nonnull %692, ptr noundef null) #24
  br label %725

694:                                              ; preds = %725
  %695 = getelementptr inbounds nuw i8, ptr %689, i64 4
  %696 = icmp eq ptr %695, %688
  br i1 %696, label %699, label %697

697:                                              ; preds = %694
  store i32 0, ptr %695, align 4, !tbaa !6
  %698 = getelementptr inbounds nuw i8, ptr %689, i64 8
  br label %729

699:                                              ; preds = %694
  %700 = ptrtoint ptr %688 to i64
  %701 = ptrtoint ptr %690 to i64
  %702 = sub i64 %700, %701
  %703 = icmp eq i64 %702, 9223372036854775804
  br i1 %703, label %704, label %706

704:                                              ; preds = %699
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.23) #27
          to label %705 unwind label %775

705:                                              ; preds = %704
  unreachable

706:                                              ; preds = %699
  %707 = ashr exact i64 %702, 2
  %708 = call i64 @llvm.umax.i64(i64 %707, i64 1)
  %709 = add nsw i64 %708, %707
  %710 = icmp ult i64 %709, %707
  %711 = call i64 @llvm.umin.i64(i64 %709, i64 2305843009213693951)
  %712 = select i1 %710, i64 2305843009213693951, i64 %711
  %713 = icmp ne i64 %712, 0
  call void @llvm.assume(i1 %713)
  %714 = shl nuw nsw i64 %712, 2
  %715 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %714) #25
          to label %716 unwind label %775

716:                                              ; preds = %706
  %717 = getelementptr inbounds i8, ptr %715, i64 %702
  store i32 0, ptr %717, align 4, !tbaa !6
  %718 = icmp sgt i64 %702, 0
  br i1 %718, label %719, label %720

719:                                              ; preds = %716
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 %715, ptr nonnull align 4 %690, i64 %702, i1 false)
  br label %720

720:                                              ; preds = %719, %716
  call void @_ZdlPvm(ptr noundef nonnull %690, i64 noundef %702) #26
  %721 = getelementptr inbounds nuw i8, ptr %717, i64 4
  %722 = getelementptr inbounds nuw i32, ptr %715, i64 %712
  br label %729

723:                                              ; preds = %665, %663
  %724 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %18) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %18) #24
  br label %868

725:                                              ; preds = %687, %725
  %726 = phi i32 [ 0, %687 ], [ %727, %725 ]
  call void @_Z9example24ss(i16 noundef 2, i16 noundef 4)
  %727 = add nuw nsw i32 %726, 1
  %728 = icmp eq i32 %727, 524288
  br i1 %728, label %694, label %725, !llvm.loop !104

729:                                              ; preds = %720, %697
  %730 = phi ptr [ %722, %720 ], [ %688, %697 ]
  %731 = phi ptr [ %721, %720 ], [ %698, %697 ]
  %732 = phi ptr [ %715, %720 ], [ %690, %697 ]
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %19) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %19) #24
  call void @_Z9example25v()
  call void @llvm.lifetime.start.p0(ptr nonnull %20) #24
  store ptr @.str.16, ptr %20, align 8, !tbaa !85
  %733 = getelementptr inbounds nuw i8, ptr %20, i64 8
  store i8 %207, ptr %733, align 8, !tbaa !91
  %734 = getelementptr inbounds nuw i8, ptr %20, i64 16
  %735 = call i32 @gettimeofday(ptr noundef nonnull %734, ptr noundef null) #24
  br label %777

736:                                              ; preds = %777, %736
  %737 = phi i32 [ %746, %736 ], [ 1, %777 ]
  %738 = phi i64 [ %747, %736 ], [ 0, %777 ]
  %739 = getelementptr inbounds nuw i8, ptr @dj, i64 %738
  %740 = mul i32 %737, 3
  %741 = load i8, ptr %739, align 1, !tbaa !81
  %742 = zext i8 %741 to i32
  %743 = xor i32 %740, %742
  %744 = lshr i32 %740, 8
  %745 = shl i32 %743, 8
  %746 = xor i32 %745, %744
  %747 = add nuw nsw i64 %738, 1
  %748 = icmp eq i64 %747, 4096
  br i1 %748, label %749, label %736, !llvm.loop !84

749:                                              ; preds = %736
  %750 = icmp eq ptr %731, %730
  br i1 %750, label %752, label %751

751:                                              ; preds = %749
  store i32 %746, ptr %731, align 4, !tbaa !6
  br label %781

752:                                              ; preds = %749
  %753 = ptrtoint ptr %730 to i64
  %754 = ptrtoint ptr %732 to i64
  %755 = sub i64 %753, %754
  %756 = icmp eq i64 %755, 9223372036854775804
  br i1 %756, label %757, label %759

757:                                              ; preds = %752
  invoke void @_ZSt20__throw_length_errorPKc(ptr noundef nonnull @.str.23) #27
          to label %758 unwind label %846

758:                                              ; preds = %757
  unreachable

759:                                              ; preds = %752
  %760 = ashr exact i64 %755, 2
  %761 = call i64 @llvm.umax.i64(i64 %760, i64 1)
  %762 = add nsw i64 %761, %760
  %763 = icmp ult i64 %762, %760
  %764 = call i64 @llvm.umin.i64(i64 %762, i64 2305843009213693951)
  %765 = select i1 %763, i64 2305843009213693951, i64 %764
  %766 = icmp ne i64 %765, 0
  call void @llvm.assume(i1 %766)
  %767 = shl nuw nsw i64 %765, 2
  %768 = invoke noalias noundef nonnull ptr @_Znwm(i64 noundef %767) #25
          to label %769 unwind label %846

769:                                              ; preds = %759
  %770 = getelementptr inbounds i8, ptr %768, i64 %755
  store i32 %746, ptr %770, align 4, !tbaa !6
  %771 = icmp sgt i64 %755, 0
  br i1 %771, label %772, label %773

772:                                              ; preds = %769
  call void @llvm.memcpy.p0.p0.i64(ptr nonnull align 4 %768, ptr nonnull align 4 %732, i64 %755, i1 false)
  br label %773

773:                                              ; preds = %772, %769
  call void @_ZdlPvm(ptr noundef nonnull %732, i64 noundef %755) #26
  %774 = getelementptr inbounds nuw i32, ptr %768, i64 %765
  br label %781

775:                                              ; preds = %706, %704
  %776 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %19) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %19) #24
  br label %868

777:                                              ; preds = %729, %777
  %778 = phi i32 [ 0, %729 ], [ %779, %777 ]
  call void @_Z9example25v()
  %779 = add nuw nsw i32 %778, 1
  %780 = icmp eq i32 %779, 524288
  br i1 %780, label %736, label %777, !llvm.loop !105

781:                                              ; preds = %773, %751
  %782 = phi ptr [ %774, %773 ], [ %730, %751 ]
  %783 = phi ptr [ %770, %773 ], [ %731, %751 ]
  %784 = phi ptr [ %768, %773 ], [ %732, %751 ]
  %785 = ptrtoint ptr %783 to i64
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %20) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %20) #24
  %786 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !106
  %787 = getelementptr i8, ptr %786, i64 -24
  %788 = load i64, ptr %787, align 8
  %789 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %788
  %790 = getelementptr inbounds nuw i8, ptr %789, i64 24
  %791 = load i32, ptr %790, align 8, !tbaa !108
  %792 = and i32 %791, -75
  %793 = or disjoint i32 %792, 8
  store i32 %793, ptr %790, align 8, !tbaa !117
  %794 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.17, i64 noundef 10)
          to label %795 unwind label %848

795:                                              ; preds = %781
  %796 = getelementptr inbounds nuw i8, ptr %783, i64 4
  %797 = ptrtoint ptr %784 to i64
  %798 = ptrtoint ptr %796 to i64
  %799 = icmp eq ptr %784, %796
  br i1 %799, label %836, label %800

800:                                              ; preds = %795
  %801 = sub i64 %785, %797
  %802 = lshr i64 %801, 2
  %803 = add nuw nsw i64 %802, 1
  %804 = icmp ult i64 %801, 28
  br i1 %804, label %826, label %805

805:                                              ; preds = %800
  %806 = and i64 %803, 9223372036854775800
  %807 = shl i64 %806, 2
  %808 = getelementptr i8, ptr %784, i64 %807
  br label %809

809:                                              ; preds = %809, %805
  %810 = phi i64 [ 0, %805 ], [ %820, %809 ]
  %811 = phi <4 x i32> [ zeroinitializer, %805 ], [ %818, %809 ]
  %812 = phi <4 x i32> [ zeroinitializer, %805 ], [ %819, %809 ]
  %813 = shl i64 %810, 2
  %814 = getelementptr i8, ptr %784, i64 %813
  %815 = getelementptr i8, ptr %814, i64 16
  %816 = load <4 x i32>, ptr %814, align 4, !tbaa !6
  %817 = load <4 x i32>, ptr %815, align 4, !tbaa !6
  %818 = add <4 x i32> %816, %811
  %819 = add <4 x i32> %817, %812
  %820 = add nuw i64 %810, 8
  %821 = icmp eq i64 %820, %806
  br i1 %821, label %822, label %809, !llvm.loop !118

822:                                              ; preds = %809
  %823 = add <4 x i32> %819, %818
  %824 = call i32 @llvm.vector.reduce.add.v4i32(<4 x i32> %823)
  %825 = icmp eq i64 %803, %806
  br i1 %825, label %836, label %826

826:                                              ; preds = %800, %822
  %827 = phi i32 [ 0, %800 ], [ %824, %822 ]
  %828 = phi ptr [ %784, %800 ], [ %808, %822 ]
  br label %829

829:                                              ; preds = %826, %829
  %830 = phi i32 [ %833, %829 ], [ %827, %826 ]
  %831 = phi ptr [ %834, %829 ], [ %828, %826 ]
  %832 = load i32, ptr %831, align 4, !tbaa !6
  %833 = add i32 %832, %830
  %834 = getelementptr inbounds nuw i8, ptr %831, i64 4
  %835 = icmp eq ptr %831, %783
  br i1 %835, label %836, label %829, !llvm.loop !119

836:                                              ; preds = %829, %822, %795
  %837 = phi i32 [ 0, %795 ], [ %824, %822 ], [ %833, %829 ]
  %838 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i32 noundef %837)
          to label %839 unwind label %848

839:                                              ; preds = %836
  %840 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %838, ptr noundef nonnull @.str.18, i64 noundef 2)
          to label %841 unwind label %848

841:                                              ; preds = %839
  %842 = sub i64 %798, %797
  %843 = ashr exact i64 %842, 2
  br i1 %799, label %844, label %850

844:                                              ; preds = %859, %841
  %845 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.20, i64 noundef 1)
          to label %865 unwind label %848

846:                                              ; preds = %759, %757
  %847 = landingpad { ptr, i32 }
          cleanup
  call void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %20) #24
  call void @llvm.lifetime.end.p0(ptr nonnull %20) #24
  br label %868

848:                                              ; preds = %844, %839, %781, %836
  %849 = landingpad { ptr, i32 }
          cleanup
  br label %868

850:                                              ; preds = %841, %859
  %851 = phi i64 [ %861, %859 ], [ 0, %841 ]
  %852 = phi i32 [ %860, %859 ], [ 0, %841 ]
  %853 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.19, i64 noundef 1)
          to label %854 unwind label %863

854:                                              ; preds = %850
  %855 = getelementptr inbounds nuw i32, ptr %784, i64 %851
  %856 = load i32, ptr %855, align 4, !tbaa !6
  %857 = zext i32 %856 to i64
  %858 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i64 noundef %857)
          to label %859 unwind label %863

859:                                              ; preds = %854
  %860 = add i32 %852, 1
  %861 = zext i32 %860 to i64
  %862 = icmp ugt i64 %843, %861
  br i1 %862, label %850, label %844, !llvm.loop !120

863:                                              ; preds = %854, %850
  %864 = landingpad { ptr, i32 }
          cleanup
  br label %868

865:                                              ; preds = %844
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #24
  %866 = ptrtoint ptr %782 to i64
  %867 = sub i64 %866, %797
  call void @_ZdlPvm(ptr noundef nonnull %784, i64 noundef %867) #26
  ret i32 0

868:                                              ; preds = %432, %526, %578, %633, %681, %723, %775, %846, %848, %863, %349, %298, %278
  %869 = phi { ptr, i32 } [ %350, %349 ], [ %299, %298 ], [ %279, %278 ], [ %433, %432 ], [ %527, %526 ], [ %579, %578 ], [ %634, %633 ], [ %682, %681 ], [ %724, %723 ], [ %776, %775 ], [ %847, %846 ], [ %864, %863 ], [ %849, %848 ]
  %870 = phi ptr [ %271, %349 ], [ %248, %298 ], [ %225, %278 ], [ %325, %432 ], [ %418, %526 ], [ %480, %578 ], [ %535, %633 ], [ %587, %681 ], [ %638, %723 ], [ %690, %775 ], [ %732, %846 ], [ %784, %863 ], [ %784, %848 ]
  %871 = phi ptr [ %306, %349 ], [ %300, %298 ], [ %280, %278 ], [ %332, %432 ], [ %421, %526 ], [ %478, %578 ], [ %533, %633 ], [ %585, %681 ], [ %636, %723 ], [ %688, %775 ], [ %730, %846 ], [ %782, %863 ], [ %782, %848 ]
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #24
  %872 = ptrtoint ptr %871 to i64
  %873 = ptrtoint ptr %870 to i64
  %874 = sub i64 %872, %873
  call void @_ZdlPvm(ptr noundef nonnull %870, i64 noundef %874) #26
  br label %875

875:                                              ; preds = %255, %868
  %876 = phi { ptr, i32 } [ %869, %868 ], [ %256, %255 ]
  resume { ptr, i32 } %876
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN5TimerD2Ev(ptr noundef nonnull align 8 dereferenceable(48) %0) unnamed_addr #12 comdat personality ptr @__gxx_personality_v0 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %3 = tail call i32 @gettimeofday(ptr noundef nonnull %2, ptr noundef null) #24
  %4 = load i64, ptr %2, align 8, !tbaa !121
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %6 = load i64, ptr %5, align 8, !tbaa !122
  %7 = sub nsw i64 %4, %6
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 40
  %9 = load i64, ptr %8, align 8, !tbaa !123
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %11 = load i64, ptr %10, align 8, !tbaa !124
  %12 = sub nsw i64 %9, %11
  %13 = mul nsw i64 %7, 1000
  %14 = sitofp i64 %13 to double
  %15 = sitofp i64 %12 to double
  %16 = fdiv double %15, 1.000000e+03
  %17 = fadd double %16, %14
  %18 = fadd double %17, 5.000000e-01
  %19 = fptosi double %18 to i64
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %21 = load i8, ptr %20, align 8, !tbaa !91, !range !125, !noundef !126
  %22 = trunc nuw i8 %21 to i1
  br i1 %22, label %23, label %43

23:                                               ; preds = %1
  %24 = load ptr, ptr %0, align 8, !tbaa !85
  %25 = icmp eq ptr %24, null
  br i1 %25, label %26, label %34

26:                                               ; preds = %23
  %27 = load ptr, ptr @_ZSt4cout, align 8, !tbaa !106
  %28 = getelementptr i8, ptr %27, i64 -24
  %29 = load i64, ptr %28, align 8
  %30 = getelementptr inbounds i8, ptr @_ZSt4cout, i64 %29
  %31 = getelementptr inbounds nuw i8, ptr %30, i64 32
  %32 = load i32, ptr %31, align 8, !tbaa !127
  %33 = or i32 %32, 1
  invoke void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264) %30, i32 noundef %33)
          to label %37 unwind label %44

34:                                               ; preds = %23
  %35 = tail call noundef i64 @strlen(ptr noundef nonnull dereferenceable(1) %24) #24
  %36 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull %24, i64 noundef %35)
          to label %37 unwind label %44

37:                                               ; preds = %26, %34
  %38 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, ptr noundef nonnull @.str.21, i64 noundef 2)
          to label %39 unwind label %44

39:                                               ; preds = %37
  %40 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIlEERSoT_(ptr noundef nonnull align 8 dereferenceable(8) @_ZSt4cout, i64 noundef %19)
          to label %41 unwind label %44

41:                                               ; preds = %39
  %42 = invoke noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8) %40, ptr noundef nonnull @.str.22, i64 noundef 7)
          to label %43 unwind label %44

43:                                               ; preds = %41, %1
  ret void

44:                                               ; preds = %41, %39, %37, %34, %26
  %45 = landingpad { ptr, i32 }
          catch ptr null
  %46 = extractvalue { ptr, i32 } %45, 0
  tail call void @__clang_call_terminate(ptr %46) #28
  unreachable
}

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSolsEi(ptr noundef nonnull align 8 dereferenceable(8), i32 noundef) local_unnamed_addr #13

; Function Attrs: nofree nounwind
declare noundef i32 @gettimeofday(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #14

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #15 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #24
  tail call void @_ZSt9terminatev() #28
  unreachable
}

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #16

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertIlEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #13

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZSt16__ostream_insertIcSt11char_traitsIcEERSt13basic_ostreamIT_T0_ES6_PKS3_l(ptr noundef nonnull align 8 dereferenceable(8), ptr noundef, i64 noundef) local_unnamed_addr #13

declare void @_ZNSt9basic_iosIcSt11char_traitsIcEE5clearESt12_Ios_Iostate(ptr noundef nonnull align 8 dereferenceable(264), i32 noundef) local_unnamed_addr #13

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #17

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #18

; Function Attrs: cold noreturn
declare void @_ZSt20__throw_length_errorPKc(ptr noundef) local_unnamed_addr #19

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #20

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #21

declare noundef nonnull align 8 dereferenceable(8) ptr @_ZNSo9_M_insertImEERSoT_(ptr noundef nonnull align 8 dereferenceable(8), i64 noundef) local_unnamed_addr #13

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #22

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #23

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umin.i64(i64, i64) #23

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v4i32(<4 x i32>) #23

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree noinline norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nofree noinline norecurse nosync nounwind memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nofree noinline norecurse nosync nounwind memory(read, argmem: write, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { mustprogress nofree noinline norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nofree norecurse nosync nounwind memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #14 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #15 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #16 = { cold nofree noreturn }
attributes #17 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #18 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #19 = { cold noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #20 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #21 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #22 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #23 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #24 = { nounwind }
attributes #25 = { builtin allocsize(0) }
attributes #26 = { builtin nounwind }
attributes #27 = { cold noreturn }
attributes #28 = { noreturn nounwind }

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
!9 = !{!"Simple C++ TBAA"}
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !12, !13}
!15 = distinct !{!15, !11, !13, !12}
!16 = distinct !{!16, !11, !12, !13}
!17 = distinct !{!17, !11, !13, !12}
!18 = distinct !{!18, !11, !12, !13}
!19 = distinct !{!19, !11, !13, !12}
!20 = distinct !{!20, !11, !12, !13}
!21 = distinct !{!21, !11, !13, !12}
!22 = distinct !{!22, !11, !12, !13}
!23 = distinct !{!23, !11, !13, !12}
!24 = distinct !{!24, !11, !12, !13}
!25 = distinct !{!25, !11, !13, !12}
!26 = distinct !{!26, !11, !12, !13}
!27 = distinct !{!27, !11, !12, !13}
!28 = distinct !{!28, !11, !12, !13}
!29 = distinct !{!29, !11, !12, !13}
!30 = distinct !{!30, !11, !12, !13}
!31 = distinct !{!31, !11, !12, !13}
!32 = distinct !{!32, !11, !12, !13}
!33 = distinct !{!33, !11, !12, !13}
!34 = distinct !{!34, !11, !12, !13}
!35 = distinct !{!35, !11, !12, !13}
!36 = distinct !{!36, !11, !12, !13}
!37 = distinct !{!37, !11, !12, !13}
!38 = distinct !{!38, !11, !12, !13}
!39 = distinct !{!39, !11, !12, !13}
!40 = distinct !{!40, !11, !12, !13}
!41 = distinct !{!41, !11, !12, !13}
!42 = distinct !{!42, !11, !12, !13}
!43 = distinct !{!43, !11, !12, !13}
!44 = distinct !{!44, !11, !12, !13}
!45 = distinct !{!45, !11, !12, !13}
!46 = distinct !{!46, !11, !12, !13}
!47 = distinct !{!47, !11, !12, !13}
!48 = distinct !{!48, !11, !12, !13}
!49 = distinct !{!49, !11, !12, !13}
!50 = distinct !{!50, !11, !12, !13}
!51 = distinct !{!51, !11, !12, !13}
!52 = distinct !{!52, !11, !12, !13}
!53 = distinct !{!53, !11, !12, !13}
!54 = distinct !{!54, !11, !12, !13}
!55 = distinct !{!55, !11, !12, !13}
!56 = distinct !{!56, !11, !12, !13}
!57 = distinct !{!57, !11, !12, !13}
!58 = distinct !{!58, !11, !12, !13}
!59 = !{!60, !60, i64 0}
!60 = !{!"short", !8, i64 0}
!61 = distinct !{!61, !11, !12, !13}
!62 = distinct !{!62, !11, !12, !13}
!63 = distinct !{!63, !11, !12, !13}
!64 = distinct !{!64, !11, !12, !13}
!65 = !{!66, !66, i64 0}
!66 = !{!"p1 int", !67, i64 0}
!67 = !{!"any pointer", !8, i64 0}
!68 = distinct !{!68, !11, !12, !13}
!69 = distinct !{!69, !11}
!70 = distinct !{!70, !11, !12, !13}
!71 = distinct !{!71, !11}
!72 = distinct !{!72, !11, !12, !13}
!73 = distinct !{!73, !11, !12, !13}
!74 = distinct !{!74, !11, !12, !13}
!75 = distinct !{!75, !11, !12, !13}
!76 = distinct !{!76, !11, !13, !12}
!77 = !{!78, !78, i64 0}
!78 = !{!"float", !8, i64 0}
!79 = distinct !{!79, !11, !12, !13}
!80 = distinct !{!80, !11, !12, !13}
!81 = !{!8, !8, i64 0}
!82 = distinct !{!82, !11}
!83 = distinct !{!83, !11}
!84 = distinct !{!84, !11}
!85 = !{!86, !87, i64 0}
!86 = !{!"_ZTS5Timer", !87, i64 0, !88, i64 8, !89, i64 16, !89, i64 32}
!87 = !{!"p1 omnipotent char", !67, i64 0}
!88 = !{!"bool", !8, i64 0}
!89 = !{!"_ZTS7timeval", !90, i64 0, !90, i64 8}
!90 = !{!"long", !8, i64 0}
!91 = !{!86, !88, i64 8}
!92 = distinct !{!92, !11}
!93 = distinct !{!93, !11}
!94 = distinct !{!94, !11}
!95 = distinct !{!95, !11}
!96 = distinct !{!96, !11}
!97 = distinct !{!97, !11}
!98 = distinct !{!98, !11}
!99 = distinct !{!99, !11}
!100 = distinct !{!100, !11}
!101 = distinct !{!101, !11}
!102 = distinct !{!102, !11}
!103 = distinct !{!103, !11}
!104 = distinct !{!104, !11}
!105 = distinct !{!105, !11}
!106 = !{!107, !107, i64 0}
!107 = !{!"vtable pointer", !9, i64 0}
!108 = !{!109, !110, i64 24}
!109 = !{!"_ZTSSt8ios_base", !90, i64 8, !90, i64 16, !110, i64 24, !111, i64 28, !111, i64 32, !112, i64 40, !113, i64 48, !8, i64 64, !7, i64 192, !114, i64 200, !115, i64 208}
!110 = !{!"_ZTSSt13_Ios_Fmtflags", !8, i64 0}
!111 = !{!"_ZTSSt12_Ios_Iostate", !8, i64 0}
!112 = !{!"p1 _ZTSNSt8ios_base14_Callback_listE", !67, i64 0}
!113 = !{!"_ZTSNSt8ios_base6_WordsE", !67, i64 0, !90, i64 8}
!114 = !{!"p1 _ZTSNSt8ios_base6_WordsE", !67, i64 0}
!115 = !{!"_ZTSSt6locale", !116, i64 0}
!116 = !{!"p1 _ZTSNSt6locale5_ImplE", !67, i64 0}
!117 = !{!110, !110, i64 0}
!118 = distinct !{!118, !11, !12, !13}
!119 = distinct !{!119, !11, !13, !12}
!120 = distinct !{!120, !11}
!121 = !{!86, !90, i64 32}
!122 = !{!86, !90, i64 16}
!123 = !{!86, !90, i64 40}
!124 = !{!86, !90, i64 24}
!125 = !{i8 0, i8 2}
!126 = !{}
!127 = !{!109, !111, i64 32}
