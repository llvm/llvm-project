; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/McGill/chomp.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/McGill/chomp.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@ncol = dso_local local_unnamed_addr global i32 0, align 4
@nrow = dso_local local_unnamed_addr global i32 0, align 4
@game_tree = dso_local local_unnamed_addr global ptr null, align 8
@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.3 = private unnamed_addr constant [14 x i8] c"  value = %d\0A\00", align 1
@wanted = dso_local local_unnamed_addr global ptr null, align 8
@.str.8 = private unnamed_addr constant [14 x i8] c" Selection : \00", align 1
@.str.13 = private unnamed_addr constant [27 x i8] c"Enter number of Columns : \00", align 1
@.str.14 = private unnamed_addr constant [27 x i8] c"Enter number of Rows    : \00", align 1
@.str.15 = private unnamed_addr constant [28 x i8] c"player %d plays at (%d,%d)\0A\00", align 1
@.str.16 = private unnamed_addr constant [17 x i8] c"player %d loses\0A\00", align 1
@stdout = external local_unnamed_addr global ptr, align 8
@str = private unnamed_addr constant [2 x i8] c")\00", align 4
@str.18 = private unnamed_addr constant [12 x i8] c"For state :\00", align 4
@str.19 = private unnamed_addr constant [19 x i8] c"We get, in order :\00", align 4
@str.20 = private unnamed_addr constant [33 x i8] c"Mode : 1 -> multiple first moves\00", align 4
@str.21 = private unnamed_addr constant [24 x i8] c"       2 -> report game\00", align 4
@str.22 = private unnamed_addr constant [27 x i8] c"       3 -> good positions\00", align 4

; Function Attrs: mustprogress nofree nounwind willreturn uwtable
define dso_local noalias noundef ptr @copy_data(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @ncol, align 4, !tbaa !6
  %3 = sext i32 %2 to i64
  %4 = shl nsw i64 %3, 2
  %5 = tail call noalias ptr @malloc(i64 noundef %4) #15
  %6 = icmp eq i32 %2, 0
  br i1 %6, label %17, label %7

7:                                                ; preds = %1
  %8 = add i32 %2, -1
  %9 = sext i32 %8 to i64
  %10 = zext i32 %8 to i64
  %11 = sub nsw i64 %9, %10
  %12 = shl nsw i64 %11, 2
  %13 = getelementptr i8, ptr %5, i64 %12
  %14 = getelementptr i8, ptr %0, i64 %12
  %15 = zext i32 %2 to i64
  %16 = shl nuw nsw i64 %15, 2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %13, ptr align 4 %14, i64 %16, i1 false), !tbaa !6
  br label %17

17:                                               ; preds = %7, %1
  ret ptr %5
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @next_data(ptr noundef captures(none) %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @ncol, align 4, !tbaa !6
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %17, label %4

4:                                                ; preds = %1, %13
  %5 = phi i32 [ %14, %13 ], [ 0, %1 ]
  %6 = zext nneg i32 %5 to i64
  %7 = getelementptr inbounds nuw i32, ptr %0, i64 %6
  %8 = load i32, ptr %7, align 4, !tbaa !6
  %9 = load i32, ptr @nrow, align 4, !tbaa !6
  %10 = icmp eq i32 %8, %9
  br i1 %10, label %13, label %11

11:                                               ; preds = %4
  %12 = add nsw i32 %8, 1
  store i32 %12, ptr %7, align 4, !tbaa !6
  br label %17

13:                                               ; preds = %4
  %14 = add nuw nsw i32 %5, 1
  store i32 0, ptr %7, align 4, !tbaa !6
  %15 = load i32, ptr @ncol, align 4, !tbaa !6
  %16 = icmp eq i32 %14, %15
  br i1 %16, label %17, label %4, !llvm.loop !10

17:                                               ; preds = %13, %11, %1
  %18 = phi i32 [ 0, %1 ], [ 1, %11 ], [ 0, %13 ]
  ret i32 %18
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @melt_data(ptr noundef captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #3 {
  %3 = load i32, ptr @ncol, align 4, !tbaa !6
  %4 = icmp eq i32 %3, 0
  br i1 %4, label %70, label %5

5:                                                ; preds = %2
  %6 = sext i32 %3 to i64
  %7 = icmp ult i32 %3, 8
  br i1 %7, label %57, label %8

8:                                                ; preds = %5
  %9 = shl nsw i64 %6, 2
  %10 = getelementptr i8, ptr %0, i64 %9
  %11 = getelementptr i8, ptr %1, i64 %9
  %12 = icmp ult ptr %0, %11
  %13 = icmp ult ptr %1, %10
  %14 = and i1 %12, %13
  br i1 %14, label %57, label %15

15:                                               ; preds = %8
  %16 = and i64 %6, -4
  %17 = and i64 %6, 3
  br label %18

18:                                               ; preds = %52, %15
  %19 = phi i64 [ 0, %15 ], [ %53, %52 ]
  %20 = sub i64 %6, %19
  %21 = add nsw i64 %20, -1
  %22 = getelementptr inbounds i32, ptr %0, i64 %21
  %23 = getelementptr inbounds i8, ptr %22, i64 -12
  %24 = load <4 x i32>, ptr %23, align 4, !tbaa !6, !alias.scope !12, !noalias !15
  %25 = getelementptr inbounds i32, ptr %1, i64 %21
  %26 = getelementptr inbounds i8, ptr %25, i64 -12
  %27 = load <4 x i32>, ptr %26, align 4, !tbaa !6, !alias.scope !15
  %28 = icmp sgt <4 x i32> %24, %27
  %29 = extractelement <4 x i1> %28, i64 3
  br i1 %29, label %30, label %34

30:                                               ; preds = %18
  %31 = getelementptr i32, ptr %0, i64 %20
  %32 = getelementptr i8, ptr %31, i64 -4
  %33 = extractelement <4 x i32> %27, i64 3
  store i32 %33, ptr %32, align 4, !tbaa !6, !alias.scope !12, !noalias !15
  br label %34

34:                                               ; preds = %30, %18
  %35 = extractelement <4 x i1> %28, i64 2
  br i1 %35, label %36, label %40

36:                                               ; preds = %34
  %37 = getelementptr i32, ptr %0, i64 %20
  %38 = getelementptr i8, ptr %37, i64 -8
  %39 = extractelement <4 x i32> %27, i64 2
  store i32 %39, ptr %38, align 4, !tbaa !6, !alias.scope !12, !noalias !15
  br label %40

40:                                               ; preds = %36, %34
  %41 = extractelement <4 x i1> %28, i64 1
  br i1 %41, label %42, label %46

42:                                               ; preds = %40
  %43 = getelementptr i32, ptr %0, i64 %20
  %44 = getelementptr i8, ptr %43, i64 -12
  %45 = extractelement <4 x i32> %27, i64 1
  store i32 %45, ptr %44, align 4, !tbaa !6, !alias.scope !12, !noalias !15
  br label %46

46:                                               ; preds = %42, %40
  %47 = extractelement <4 x i1> %28, i64 0
  br i1 %47, label %48, label %52

48:                                               ; preds = %46
  %49 = getelementptr i32, ptr %0, i64 %20
  %50 = getelementptr i8, ptr %49, i64 -16
  %51 = extractelement <4 x i32> %27, i64 0
  store i32 %51, ptr %50, align 4, !tbaa !6, !alias.scope !12, !noalias !15
  br label %52

52:                                               ; preds = %48, %46
  %53 = add nuw i64 %19, 4
  %54 = icmp eq i64 %53, %16
  br i1 %54, label %55, label %18, !llvm.loop !17

55:                                               ; preds = %52
  %56 = icmp eq i64 %16, %6
  br i1 %56, label %70, label %57

57:                                               ; preds = %8, %5, %55
  %58 = phi i64 [ %6, %8 ], [ %6, %5 ], [ %17, %55 ]
  br label %59

59:                                               ; preds = %57, %68
  %60 = phi i64 [ %61, %68 ], [ %58, %57 ]
  %61 = add nsw i64 %60, -1
  %62 = getelementptr inbounds i32, ptr %0, i64 %61
  %63 = load i32, ptr %62, align 4, !tbaa !6
  %64 = getelementptr inbounds i32, ptr %1, i64 %61
  %65 = load i32, ptr %64, align 4, !tbaa !6
  %66 = icmp sgt i32 %63, %65
  br i1 %66, label %67, label %68

67:                                               ; preds = %59
  store i32 %65, ptr %62, align 4, !tbaa !6
  br label %68

68:                                               ; preds = %67, %59
  %69 = icmp eq i64 %61, 0
  br i1 %69, label %70, label %59, !llvm.loop !20

70:                                               ; preds = %68, %55, %2
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @equal_data(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #4 {
  %3 = load i32, ptr @ncol, align 4, !tbaa !6
  %4 = sext i32 %3 to i64
  br label %5

5:                                                ; preds = %8, %2
  %6 = phi i64 [ %9, %8 ], [ %4, %2 ]
  %7 = icmp eq i64 %6, 0
  br i1 %7, label %18, label %8

8:                                                ; preds = %5
  %9 = add nsw i64 %6, -1
  %10 = getelementptr inbounds i32, ptr %0, i64 %9
  %11 = load i32, ptr %10, align 4, !tbaa !6
  %12 = getelementptr inbounds i32, ptr %1, i64 %9
  %13 = load i32, ptr %12, align 4, !tbaa !6
  %14 = icmp eq i32 %11, %13
  br i1 %14, label %5, label %15, !llvm.loop !21

15:                                               ; preds = %8
  %16 = icmp slt i64 %6, 1
  %17 = zext i1 %16 to i32
  br label %18

18:                                               ; preds = %5, %15
  %19 = phi i32 [ %17, %15 ], [ 1, %5 ]
  ret i32 %19
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @valid_data(ptr noundef readonly captures(none) %0) local_unnamed_addr #4 {
  %2 = load i32, ptr @ncol, align 4, !tbaa !6
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %16, label %4

4:                                                ; preds = %1
  %5 = load i32, ptr @nrow, align 4, !tbaa !6
  %6 = zext i32 %2 to i64
  br label %10

7:                                                ; preds = %10
  %8 = add nuw nsw i64 %11, 1
  %9 = icmp eq i64 %8, %6
  br i1 %9, label %16, label %10, !llvm.loop !22

10:                                               ; preds = %4, %7
  %11 = phi i64 [ 0, %4 ], [ %8, %7 ]
  %12 = phi i32 [ %5, %4 ], [ %14, %7 ]
  %13 = getelementptr inbounds nuw i32, ptr %0, i64 %11
  %14 = load i32, ptr %13, align 4, !tbaa !6
  %15 = icmp sgt i32 %14, %12
  br i1 %15, label %16, label %7

16:                                               ; preds = %7, %10, %1
  %17 = phi i32 [ 1, %1 ], [ 0, %10 ], [ 1, %7 ]
  ret i32 %17
}

; Function Attrs: nounwind uwtable
define dso_local void @dump_list(ptr noundef captures(address_is_null) %0) local_unnamed_addr #5 {
  %2 = icmp eq ptr %0, null
  br i1 %2, label %3, label %4

3:                                                ; preds = %1, %4
  ret void

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %6 = load ptr, ptr %5, align 8, !tbaa !23
  tail call void @dump_list(ptr noundef %6)
  %7 = load ptr, ptr %0, align 8, !tbaa !28
  tail call void @free(ptr noundef %7) #16
  tail call void @free(ptr noundef nonnull %0) #16
  br label %3
}

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #6

; Function Attrs: nounwind uwtable
define dso_local void @dump_play(ptr noundef captures(address_is_null) %0) local_unnamed_addr #5 {
  %2 = icmp eq ptr %0, null
  br i1 %2, label %3, label %4

3:                                                ; preds = %1, %4
  ret void

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %6 = load ptr, ptr %5, align 8, !tbaa !29
  tail call void @dump_play(ptr noundef %6)
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %8 = load ptr, ptr %7, align 8, !tbaa !32
  tail call void @dump_list(ptr noundef %8)
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load ptr, ptr %9, align 8, !tbaa !33
  tail call void @free(ptr noundef %10) #16
  tail call void @free(ptr noundef nonnull %0) #16
  br label %3
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable
define dso_local i32 @get_value(ptr noundef readonly captures(none) %0) local_unnamed_addr #4 {
  %2 = load i32, ptr @ncol, align 4, !tbaa !6
  %3 = sext i32 %2 to i64
  br label %4

4:                                                ; preds = %21, %1
  %5 = phi ptr [ @game_tree, %1 ], [ %22, %21 ]
  %6 = load ptr, ptr %5, align 8, !tbaa !34
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !33
  br label %9

9:                                                ; preds = %12, %4
  %10 = phi i64 [ %13, %12 ], [ %3, %4 ]
  %11 = icmp eq i64 %10, 0
  br i1 %11, label %23, label %12

12:                                               ; preds = %9
  %13 = add nsw i64 %10, -1
  %14 = getelementptr inbounds i32, ptr %8, i64 %13
  %15 = load i32, ptr %14, align 4, !tbaa !6
  %16 = getelementptr inbounds i32, ptr %0, i64 %13
  %17 = load i32, ptr %16, align 4, !tbaa !6
  %18 = icmp eq i32 %15, %17
  br i1 %18, label %9, label %19, !llvm.loop !21

19:                                               ; preds = %12
  %20 = icmp sgt i64 %10, 0
  br i1 %20, label %21, label %23

21:                                               ; preds = %19
  %22 = getelementptr inbounds nuw i8, ptr %6, i64 24
  br label %4, !llvm.loop !35

23:                                               ; preds = %19, %9
  %24 = load i32, ptr %6, align 8, !tbaa !36
  ret i32 %24
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @show_data(ptr noundef readonly captures(none) %0) local_unnamed_addr #7 {
  %2 = load i32, ptr @ncol, align 4, !tbaa !6
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %19, label %4

4:                                                ; preds = %1, %13
  %5 = phi i64 [ %6, %13 ], [ 0, %1 ]
  %6 = add nuw nsw i64 %5, 1
  %7 = getelementptr inbounds nuw i32, ptr %0, i64 %5
  %8 = load i32, ptr %7, align 4, !tbaa !6
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %8)
  %10 = load i32, ptr @ncol, align 4, !tbaa !6
  %11 = zext i32 %10 to i64
  %12 = icmp eq i64 %6, %11
  br i1 %12, label %19, label %13

13:                                               ; preds = %4
  %14 = load ptr, ptr @stdout, align 8, !tbaa !37
  %15 = tail call i32 @putc(i32 noundef 44, ptr noundef %14)
  %16 = load i32, ptr @ncol, align 4, !tbaa !6
  %17 = zext i32 %16 to i64
  %18 = icmp eq i64 %6, %17
  br i1 %18, label %19, label %4, !llvm.loop !39

19:                                               ; preds = %4, %13, %1
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #8

; Function Attrs: nofree nounwind uwtable
define dso_local void @show_move(ptr noundef readonly captures(none) %0) local_unnamed_addr #7 {
  %2 = load ptr, ptr @stdout, align 8, !tbaa !37
  %3 = tail call i32 @putc(i32 noundef 40, ptr noundef %2)
  %4 = load i32, ptr @ncol, align 4, !tbaa !6
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %21, label %6

6:                                                ; preds = %1, %15
  %7 = phi i64 [ %8, %15 ], [ 0, %1 ]
  %8 = add nuw nsw i64 %7, 1
  %9 = getelementptr inbounds nuw i32, ptr %0, i64 %7
  %10 = load i32, ptr %9, align 4, !tbaa !6
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %10)
  %12 = load i32, ptr @ncol, align 4, !tbaa !6
  %13 = zext i32 %12 to i64
  %14 = icmp eq i64 %8, %13
  br i1 %14, label %21, label %15

15:                                               ; preds = %6
  %16 = load ptr, ptr @stdout, align 8, !tbaa !37
  %17 = tail call i32 @putc(i32 noundef 44, ptr noundef %16)
  %18 = load i32, ptr @ncol, align 4, !tbaa !6
  %19 = zext i32 %18 to i64
  %20 = icmp eq i64 %8, %19
  br i1 %20, label %21, label %6, !llvm.loop !39

21:                                               ; preds = %6, %15, %1
  %22 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @show_list(ptr noundef readonly captures(address_is_null) %0) local_unnamed_addr #7 {
  %2 = icmp eq ptr %0, null
  br i1 %2, label %30, label %3

3:                                                ; preds = %1, %25
  %4 = phi ptr [ %28, %25 ], [ %0, %1 ]
  %5 = load ptr, ptr %4, align 8, !tbaa !28
  %6 = load ptr, ptr @stdout, align 8, !tbaa !37
  %7 = tail call i32 @putc(i32 noundef 40, ptr noundef %6)
  %8 = load i32, ptr @ncol, align 4, !tbaa !6
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %25, label %10

10:                                               ; preds = %3, %19
  %11 = phi i64 [ %12, %19 ], [ 0, %3 ]
  %12 = add nuw nsw i64 %11, 1
  %13 = getelementptr inbounds nuw i32, ptr %5, i64 %11
  %14 = load i32, ptr %13, align 4, !tbaa !6
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %14)
  %16 = load i32, ptr @ncol, align 4, !tbaa !6
  %17 = zext i32 %16 to i64
  %18 = icmp eq i64 %12, %17
  br i1 %18, label %25, label %19

19:                                               ; preds = %10
  %20 = load ptr, ptr @stdout, align 8, !tbaa !37
  %21 = tail call i32 @putc(i32 noundef 44, ptr noundef %20)
  %22 = load i32, ptr @ncol, align 4, !tbaa !6
  %23 = zext i32 %22 to i64
  %24 = icmp eq i64 %12, %23
  br i1 %24, label %25, label %10, !llvm.loop !39

25:                                               ; preds = %10, %19, %3
  %26 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %27 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %28 = load ptr, ptr %27, align 8, !tbaa !23
  %29 = icmp eq ptr %28, null
  br i1 %29, label %30, label %3, !llvm.loop !40

30:                                               ; preds = %25, %1
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @show_play(ptr noundef readonly captures(address_is_null) %0) local_unnamed_addr #7 {
  %2 = icmp eq ptr %0, null
  br i1 %2, label %63, label %3

3:                                                ; preds = %1, %59
  %4 = phi ptr [ %61, %59 ], [ %0, %1 ]
  %5 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.18)
  %6 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %7 = load ptr, ptr %6, align 8, !tbaa !33
  %8 = load i32, ptr @ncol, align 4, !tbaa !6
  %9 = icmp eq i32 %8, 0
  br i1 %9, label %25, label %10

10:                                               ; preds = %3, %19
  %11 = phi i64 [ %12, %19 ], [ 0, %3 ]
  %12 = add nuw nsw i64 %11, 1
  %13 = getelementptr inbounds nuw i32, ptr %7, i64 %11
  %14 = load i32, ptr %13, align 4, !tbaa !6
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %14)
  %16 = load i32, ptr @ncol, align 4, !tbaa !6
  %17 = zext i32 %16 to i64
  %18 = icmp eq i64 %12, %17
  br i1 %18, label %25, label %19

19:                                               ; preds = %10
  %20 = load ptr, ptr @stdout, align 8, !tbaa !37
  %21 = tail call i32 @putc(i32 noundef 44, ptr noundef %20)
  %22 = load i32, ptr @ncol, align 4, !tbaa !6
  %23 = zext i32 %22 to i64
  %24 = icmp eq i64 %12, %23
  br i1 %24, label %25, label %10, !llvm.loop !39

25:                                               ; preds = %10, %19, %3
  %26 = load i32, ptr %4, align 8, !tbaa !36
  %27 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %26)
  %28 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.19)
  %29 = getelementptr inbounds nuw i8, ptr %4, i64 16
  %30 = load ptr, ptr %29, align 8, !tbaa !32
  %31 = icmp eq ptr %30, null
  br i1 %31, label %59, label %32

32:                                               ; preds = %25, %54
  %33 = phi ptr [ %57, %54 ], [ %30, %25 ]
  %34 = load ptr, ptr %33, align 8, !tbaa !28
  %35 = load ptr, ptr @stdout, align 8, !tbaa !37
  %36 = tail call i32 @putc(i32 noundef 40, ptr noundef %35)
  %37 = load i32, ptr @ncol, align 4, !tbaa !6
  %38 = icmp eq i32 %37, 0
  br i1 %38, label %54, label %39

39:                                               ; preds = %32, %48
  %40 = phi i64 [ %41, %48 ], [ 0, %32 ]
  %41 = add nuw nsw i64 %40, 1
  %42 = getelementptr inbounds nuw i32, ptr %34, i64 %40
  %43 = load i32, ptr %42, align 4, !tbaa !6
  %44 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %43)
  %45 = load i32, ptr @ncol, align 4, !tbaa !6
  %46 = zext i32 %45 to i64
  %47 = icmp eq i64 %41, %46
  br i1 %47, label %54, label %48

48:                                               ; preds = %39
  %49 = load ptr, ptr @stdout, align 8, !tbaa !37
  %50 = tail call i32 @putc(i32 noundef 44, ptr noundef %49)
  %51 = load i32, ptr @ncol, align 4, !tbaa !6
  %52 = zext i32 %51 to i64
  %53 = icmp eq i64 %41, %52
  br i1 %53, label %54, label %39, !llvm.loop !39

54:                                               ; preds = %48, %39, %32
  %55 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %56 = getelementptr inbounds nuw i8, ptr %33, i64 8
  %57 = load ptr, ptr %56, align 8, !tbaa !23
  %58 = icmp eq ptr %57, null
  br i1 %58, label %59, label %32, !llvm.loop !40

59:                                               ; preds = %54, %25
  %60 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %61 = load ptr, ptr %60, align 8, !tbaa !29
  %62 = icmp eq ptr %61, null
  br i1 %62, label %63, label %3, !llvm.loop !41

63:                                               ; preds = %59, %1
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @in_wanted(ptr noundef readonly captures(none) %0) local_unnamed_addr #4 {
  %2 = load ptr, ptr @wanted, align 8, !tbaa !42
  %3 = icmp eq ptr %2, null
  br i1 %3, label %26, label %4

4:                                                ; preds = %1
  %5 = load i32, ptr @ncol, align 4, !tbaa !6
  %6 = sext i32 %5 to i64
  br label %11

7:                                                ; preds = %24
  %8 = getelementptr inbounds nuw i8, ptr %12, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !42
  %10 = icmp eq ptr %9, null
  br i1 %10, label %26, label %11, !llvm.loop !43

11:                                               ; preds = %4, %7
  %12 = phi ptr [ %2, %4 ], [ %9, %7 ]
  %13 = load ptr, ptr %12, align 8, !tbaa !28
  br label %14

14:                                               ; preds = %17, %11
  %15 = phi i64 [ %18, %17 ], [ %6, %11 ]
  %16 = icmp eq i64 %15, 0
  br i1 %16, label %26, label %17

17:                                               ; preds = %14
  %18 = add nsw i64 %15, -1
  %19 = getelementptr inbounds i32, ptr %13, i64 %18
  %20 = load i32, ptr %19, align 4, !tbaa !6
  %21 = getelementptr inbounds i32, ptr %0, i64 %18
  %22 = load i32, ptr %21, align 4, !tbaa !6
  %23 = icmp eq i32 %20, %22
  br i1 %23, label %14, label %24, !llvm.loop !21

24:                                               ; preds = %17
  %25 = icmp sgt i64 %15, 0
  br i1 %25, label %7, label %26

26:                                               ; preds = %7, %24, %14, %1
  %27 = phi i32 [ 0, %1 ], [ 1, %14 ], [ 0, %7 ], [ 1, %24 ]
  ret i32 %27
}

; Function Attrs: nofree nounwind memory(readwrite, argmem: none) uwtable
define dso_local noalias noundef ptr @make_data(i32 noundef %0, i32 noundef %1) local_unnamed_addr #9 {
  %3 = load i32, ptr @ncol, align 4, !tbaa !6
  %4 = sext i32 %3 to i64
  %5 = shl nsw i64 %4, 2
  %6 = tail call noalias ptr @malloc(i64 noundef %5) #15
  %7 = icmp eq i32 %1, 0
  br i1 %7, label %26, label %8

8:                                                ; preds = %2
  %9 = load i32, ptr @nrow, align 4, !tbaa !6
  %10 = zext i32 %1 to i64
  %11 = icmp ult i32 %1, 8
  br i1 %11, label %24, label %12

12:                                               ; preds = %8
  %13 = and i64 %10, 4294967288
  %14 = insertelement <4 x i32> poison, i32 %9, i64 0
  %15 = shufflevector <4 x i32> %14, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %16

16:                                               ; preds = %16, %12
  %17 = phi i64 [ 0, %12 ], [ %20, %16 ]
  %18 = getelementptr inbounds nuw i32, ptr %6, i64 %17
  %19 = getelementptr inbounds nuw i8, ptr %18, i64 16
  store <4 x i32> %15, ptr %18, align 4, !tbaa !6
  store <4 x i32> %15, ptr %19, align 4, !tbaa !6
  %20 = add nuw i64 %17, 8
  %21 = icmp eq i64 %20, %13
  br i1 %21, label %22, label %16, !llvm.loop !44

22:                                               ; preds = %16
  %23 = icmp eq i64 %13, %10
  br i1 %23, label %26, label %24

24:                                               ; preds = %8, %22
  %25 = phi i64 [ 0, %8 ], [ %13, %22 ]
  br label %49

26:                                               ; preds = %49, %22, %2
  %27 = icmp eq i32 %1, %3
  br i1 %27, label %59, label %28

28:                                               ; preds = %26
  %29 = zext i32 %1 to i64
  %30 = zext i32 %3 to i64
  %31 = sub nsw i64 %30, %29
  %32 = icmp ult i64 %31, 8
  br i1 %32, label %47, label %33

33:                                               ; preds = %28
  %34 = and i64 %31, -8
  %35 = add nsw i64 %34, %29
  %36 = insertelement <4 x i32> poison, i32 %0, i64 0
  %37 = shufflevector <4 x i32> %36, <4 x i32> poison, <4 x i32> zeroinitializer
  %38 = getelementptr i32, ptr %6, i64 %29
  br label %39

39:                                               ; preds = %39, %33
  %40 = phi i64 [ 0, %33 ], [ %43, %39 ]
  %41 = getelementptr i32, ptr %38, i64 %40
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 16
  store <4 x i32> %37, ptr %41, align 4, !tbaa !6
  store <4 x i32> %37, ptr %42, align 4, !tbaa !6
  %43 = add nuw i64 %40, 8
  %44 = icmp eq i64 %43, %34
  br i1 %44, label %45, label %39, !llvm.loop !45

45:                                               ; preds = %39
  %46 = icmp eq i64 %31, %34
  br i1 %46, label %59, label %47

47:                                               ; preds = %28, %45
  %48 = phi i64 [ %29, %28 ], [ %35, %45 ]
  br label %54

49:                                               ; preds = %24, %49
  %50 = phi i64 [ %52, %49 ], [ %25, %24 ]
  %51 = getelementptr inbounds nuw i32, ptr %6, i64 %50
  store i32 %9, ptr %51, align 4, !tbaa !6
  %52 = add nuw nsw i64 %50, 1
  %53 = icmp eq i64 %52, %10
  br i1 %53, label %26, label %49, !llvm.loop !46

54:                                               ; preds = %47, %54
  %55 = phi i64 [ %57, %54 ], [ %48, %47 ]
  %56 = getelementptr inbounds nuw i32, ptr %6, i64 %55
  store i32 %0, ptr %56, align 4, !tbaa !6
  %57 = add nuw nsw i64 %55, 1
  %58 = icmp eq i64 %57, %30
  br i1 %58, label %59, label %54, !llvm.loop !47

59:                                               ; preds = %54, %45, %26
  ret ptr %6
}

; Function Attrs: nounwind uwtable
define dso_local ptr @make_list(ptr noundef readonly captures(none) %0, ptr noundef captures(none) initializes((0, 4)) %1, ptr noundef captures(none) %2) local_unnamed_addr #5 {
  store i32 1, ptr %1, align 4, !tbaa !6
  %4 = tail call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #15
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store ptr null, ptr %5, align 8, !tbaa !23
  %6 = load i32, ptr @nrow, align 4, !tbaa !6
  %7 = icmp eq i32 %6, 0
  %8 = load i32, ptr @ncol, align 4
  %9 = icmp eq i32 %8, 0
  %10 = select i1 %7, i1 true, i1 %9
  br i1 %10, label %218, label %11

11:                                               ; preds = %3, %211
  %12 = phi i32 [ %212, %211 ], [ %6, %3 ]
  %13 = phi i32 [ %213, %211 ], [ %8, %3 ]
  %14 = phi ptr [ %215, %211 ], [ %4, %3 ]
  %15 = phi i32 [ %216, %211 ], [ 0, %3 ]
  %16 = icmp eq i32 %13, 0
  br i1 %16, label %211, label %17

17:                                               ; preds = %11
  %18 = load ptr, ptr @wanted, align 8
  %19 = icmp eq ptr %18, null
  br label %20

20:                                               ; preds = %17, %202
  %21 = phi i32 [ %13, %17 ], [ %207, %202 ]
  %22 = phi ptr [ %14, %17 ], [ %205, %202 ]
  %23 = phi i32 [ %15, %17 ], [ %204, %202 ]
  %24 = phi i32 [ 0, %17 ], [ %206, %202 ]
  %25 = sext i32 %21 to i64
  %26 = shl nsw i64 %25, 2
  %27 = tail call noalias ptr @malloc(i64 noundef %26) #15
  %28 = icmp eq i32 %24, 0
  br i1 %28, label %47, label %29

29:                                               ; preds = %20
  %30 = load i32, ptr @nrow, align 4, !tbaa !6
  %31 = zext i32 %24 to i64
  %32 = icmp ult i32 %24, 8
  br i1 %32, label %45, label %33

33:                                               ; preds = %29
  %34 = and i64 %31, 4294967288
  %35 = insertelement <4 x i32> poison, i32 %30, i64 0
  %36 = shufflevector <4 x i32> %35, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %37

37:                                               ; preds = %37, %33
  %38 = phi i64 [ 0, %33 ], [ %41, %37 ]
  %39 = getelementptr inbounds nuw i32, ptr %27, i64 %38
  %40 = getelementptr inbounds nuw i8, ptr %39, i64 16
  store <4 x i32> %36, ptr %39, align 4, !tbaa !6
  store <4 x i32> %36, ptr %40, align 4, !tbaa !6
  %41 = add nuw i64 %38, 8
  %42 = icmp eq i64 %41, %34
  br i1 %42, label %43, label %37, !llvm.loop !48

43:                                               ; preds = %37
  %44 = icmp eq i64 %34, %31
  br i1 %44, label %47, label %45

45:                                               ; preds = %29, %43
  %46 = phi i64 [ 0, %29 ], [ %34, %43 ]
  br label %68

47:                                               ; preds = %68, %43, %20
  %48 = phi i64 [ 0, %20 ], [ %31, %43 ], [ %31, %68 ]
  %49 = zext i32 %21 to i64
  %50 = sub nsw i64 %49, %48
  %51 = icmp ult i64 %50, 8
  br i1 %51, label %66, label %52

52:                                               ; preds = %47
  %53 = and i64 %50, -8
  %54 = add nsw i64 %48, %53
  %55 = insertelement <4 x i32> poison, i32 %23, i64 0
  %56 = shufflevector <4 x i32> %55, <4 x i32> poison, <4 x i32> zeroinitializer
  %57 = getelementptr i32, ptr %27, i64 %48
  br label %58

58:                                               ; preds = %58, %52
  %59 = phi i64 [ 0, %52 ], [ %62, %58 ]
  %60 = getelementptr i32, ptr %57, i64 %59
  %61 = getelementptr inbounds nuw i8, ptr %60, i64 16
  store <4 x i32> %56, ptr %60, align 4, !tbaa !6
  store <4 x i32> %56, ptr %61, align 4, !tbaa !6
  %62 = add nuw i64 %59, 8
  %63 = icmp eq i64 %62, %53
  br i1 %63, label %64, label %58, !llvm.loop !49

64:                                               ; preds = %58
  %65 = icmp eq i64 %50, %53
  br i1 %65, label %78, label %66

66:                                               ; preds = %47, %64
  %67 = phi i64 [ %48, %47 ], [ %54, %64 ]
  br label %73

68:                                               ; preds = %45, %68
  %69 = phi i64 [ %71, %68 ], [ %46, %45 ]
  %70 = getelementptr inbounds nuw i32, ptr %27, i64 %69
  store i32 %30, ptr %70, align 4, !tbaa !6
  %71 = add nuw nsw i64 %69, 1
  %72 = icmp eq i64 %71, %31
  br i1 %72, label %47, label %68, !llvm.loop !50

73:                                               ; preds = %66, %73
  %74 = phi i64 [ %76, %73 ], [ %67, %66 ]
  %75 = getelementptr inbounds nuw i32, ptr %27, i64 %74
  store i32 %23, ptr %75, align 4, !tbaa !6
  %76 = add nuw nsw i64 %74, 1
  %77 = icmp eq i64 %76, %49
  br i1 %77, label %78, label %73, !llvm.loop !51

78:                                               ; preds = %73, %64
  %79 = icmp ult i32 %21, 8
  br i1 %79, label %103, label %80

80:                                               ; preds = %78
  %81 = and i64 %25, -8
  %82 = and i64 %25, 7
  br label %83

83:                                               ; preds = %83, %80
  %84 = phi i64 [ 0, %80 ], [ %99, %83 ]
  %85 = xor i64 %84, -1
  %86 = add i64 %85, %25
  %87 = getelementptr inbounds i32, ptr %27, i64 %86
  %88 = getelementptr inbounds i8, ptr %87, i64 -12
  %89 = getelementptr inbounds i8, ptr %87, i64 -28
  %90 = load <4 x i32>, ptr %88, align 4, !tbaa !6
  %91 = load <4 x i32>, ptr %89, align 4, !tbaa !6
  %92 = getelementptr inbounds i32, ptr %0, i64 %86
  %93 = getelementptr inbounds i8, ptr %92, i64 -12
  %94 = getelementptr inbounds i8, ptr %92, i64 -28
  %95 = load <4 x i32>, ptr %93, align 4, !tbaa !6
  %96 = load <4 x i32>, ptr %94, align 4, !tbaa !6
  %97 = tail call <4 x i32> @llvm.smin.v4i32(<4 x i32> %90, <4 x i32> %95)
  %98 = tail call <4 x i32> @llvm.smin.v4i32(<4 x i32> %91, <4 x i32> %96)
  store <4 x i32> %97, ptr %88, align 4
  store <4 x i32> %98, ptr %89, align 4
  %99 = add nuw i64 %84, 8
  %100 = icmp eq i64 %99, %81
  br i1 %100, label %101, label %83, !llvm.loop !52

101:                                              ; preds = %83
  %102 = icmp eq i64 %81, %25
  br i1 %102, label %114, label %103

103:                                              ; preds = %78, %101
  %104 = phi i64 [ %25, %78 ], [ %82, %101 ]
  br label %105

105:                                              ; preds = %103, %105
  %106 = phi i64 [ %107, %105 ], [ %104, %103 ]
  %107 = add nsw i64 %106, -1
  %108 = getelementptr inbounds i32, ptr %27, i64 %107
  %109 = load i32, ptr %108, align 4, !tbaa !6
  %110 = getelementptr inbounds i32, ptr %0, i64 %107
  %111 = load i32, ptr %110, align 4, !tbaa !6
  %112 = tail call i32 @llvm.smin.i32(i32 %109, i32 %111)
  store i32 %112, ptr %108, align 4
  %113 = icmp eq i64 %107, 0
  br i1 %113, label %114, label %105, !llvm.loop !53

114:                                              ; preds = %105, %101
  br label %115

115:                                              ; preds = %114, %118
  %116 = phi i64 [ %119, %118 ], [ %25, %114 ]
  %117 = icmp eq i64 %116, 0
  br i1 %117, label %197, label %118

118:                                              ; preds = %115
  %119 = add nsw i64 %116, -1
  %120 = getelementptr inbounds i32, ptr %27, i64 %119
  %121 = load i32, ptr %120, align 4, !tbaa !6
  %122 = getelementptr inbounds i32, ptr %0, i64 %119
  %123 = load i32, ptr %122, align 4, !tbaa !6
  %124 = icmp eq i32 %121, %123
  br i1 %124, label %115, label %125, !llvm.loop !21

125:                                              ; preds = %118
  %126 = icmp sgt i64 %116, 0
  br i1 %126, label %127, label %197

127:                                              ; preds = %125
  %128 = tail call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #15
  %129 = getelementptr inbounds nuw i8, ptr %22, i64 8
  store ptr %128, ptr %129, align 8, !tbaa !23
  %130 = tail call noalias ptr @malloc(i64 noundef %26) #15
  %131 = add i32 %21, -1
  %132 = sext i32 %131 to i64
  %133 = zext i32 %131 to i64
  %134 = sub nsw i64 %132, %133
  %135 = shl nsw i64 %134, 2
  %136 = getelementptr i8, ptr %130, i64 %135
  %137 = getelementptr i8, ptr %27, i64 %135
  %138 = shl nuw nsw i64 %49, 2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %136, ptr readonly align 4 %137, i64 %138, i1 false), !tbaa !6
  store ptr %130, ptr %128, align 8, !tbaa !28
  %139 = getelementptr inbounds nuw i8, ptr %128, i64 8
  store ptr null, ptr %139, align 8, !tbaa !23
  %140 = load ptr, ptr %129, align 8, !tbaa !23
  %141 = load i32, ptr %1, align 4, !tbaa !6
  %142 = icmp eq i32 %141, 1
  br i1 %142, label %143, label %164

143:                                              ; preds = %127, %160
  %144 = phi ptr [ %161, %160 ], [ @game_tree, %127 ]
  %145 = load ptr, ptr %144, align 8, !tbaa !34
  %146 = getelementptr inbounds nuw i8, ptr %145, i64 8
  %147 = load ptr, ptr %146, align 8, !tbaa !33
  br label %148

148:                                              ; preds = %151, %143
  %149 = phi i64 [ %152, %151 ], [ %25, %143 ]
  %150 = icmp eq i64 %149, 0
  br i1 %150, label %162, label %151

151:                                              ; preds = %148
  %152 = add nsw i64 %149, -1
  %153 = getelementptr inbounds i32, ptr %147, i64 %152
  %154 = load i32, ptr %153, align 4, !tbaa !6
  %155 = getelementptr inbounds i32, ptr %27, i64 %152
  %156 = load i32, ptr %155, align 4, !tbaa !6
  %157 = icmp eq i32 %154, %156
  br i1 %157, label %148, label %158, !llvm.loop !21

158:                                              ; preds = %151
  %159 = icmp sgt i64 %149, 0
  br i1 %159, label %160, label %162

160:                                              ; preds = %158
  %161 = getelementptr inbounds nuw i8, ptr %145, i64 24
  br label %143, !llvm.loop !35

162:                                              ; preds = %158, %148
  %163 = load i32, ptr %145, align 8, !tbaa !36
  store i32 %163, ptr %1, align 4, !tbaa !6
  br label %164

164:                                              ; preds = %162, %127
  %165 = phi i32 [ %163, %162 ], [ %141, %127 ]
  %166 = load i32, ptr %2, align 4, !tbaa !6
  %167 = icmp eq i32 %166, 0
  %168 = icmp eq i32 %165, 0
  %169 = select i1 %167, i1 %168, i1 false
  br i1 %169, label %170, label %202

170:                                              ; preds = %164
  %171 = load i32, ptr @ncol, align 4, !tbaa !6
  %172 = add nsw i32 %171, -1
  %173 = load i32, ptr @nrow, align 4, !tbaa !6
  %174 = add nsw i32 %173, -1
  br i1 %19, label %202, label %175

175:                                              ; preds = %170
  %176 = sext i32 %171 to i64
  br label %181

177:                                              ; preds = %194
  %178 = getelementptr inbounds nuw i8, ptr %182, i64 8
  %179 = load ptr, ptr %178, align 8, !tbaa !42
  %180 = icmp eq ptr %179, null
  br i1 %180, label %202, label %181, !llvm.loop !43

181:                                              ; preds = %177, %175
  %182 = phi ptr [ %18, %175 ], [ %179, %177 ]
  %183 = load ptr, ptr %182, align 8, !tbaa !28
  br label %184

184:                                              ; preds = %187, %181
  %185 = phi i64 [ %188, %187 ], [ %176, %181 ]
  %186 = icmp eq i64 %185, 0
  br i1 %186, label %196, label %187

187:                                              ; preds = %184
  %188 = add nsw i64 %185, -1
  %189 = getelementptr inbounds i32, ptr %183, i64 %188
  %190 = load i32, ptr %189, align 4, !tbaa !6
  %191 = getelementptr inbounds i32, ptr %27, i64 %188
  %192 = load i32, ptr %191, align 4, !tbaa !6
  %193 = icmp eq i32 %190, %192
  br i1 %193, label %184, label %194, !llvm.loop !21

194:                                              ; preds = %187
  %195 = icmp sgt i64 %185, 0
  br i1 %195, label %177, label %196

196:                                              ; preds = %194, %184
  store i32 2, ptr %2, align 4, !tbaa !6
  br label %202

197:                                              ; preds = %115, %125
  %198 = load i32, ptr @nrow, align 4
  %199 = add nsw i32 %198, -1
  %200 = select i1 %28, i32 %199, i32 %23
  %201 = add nsw i32 %21, -1
  br label %202

202:                                              ; preds = %177, %170, %164, %196, %197
  %203 = phi i32 [ %201, %197 ], [ %24, %164 ], [ %172, %196 ], [ %172, %170 ], [ %172, %177 ]
  %204 = phi i32 [ %200, %197 ], [ %23, %164 ], [ %174, %196 ], [ %174, %170 ], [ %174, %177 ]
  %205 = phi ptr [ %22, %197 ], [ %140, %164 ], [ %140, %196 ], [ %140, %170 ], [ %140, %177 ]
  tail call void @free(ptr noundef %27) #16
  %206 = add nsw i32 %203, 1
  %207 = load i32, ptr @ncol, align 4, !tbaa !6
  %208 = icmp eq i32 %206, %207
  br i1 %208, label %209, label %20, !llvm.loop !54

209:                                              ; preds = %202
  %210 = load i32, ptr @nrow, align 4, !tbaa !6
  br label %211

211:                                              ; preds = %209, %11
  %212 = phi i32 [ %12, %11 ], [ %210, %209 ]
  %213 = phi i32 [ 0, %11 ], [ %206, %209 ]
  %214 = phi i32 [ %15, %11 ], [ %204, %209 ]
  %215 = phi ptr [ %14, %11 ], [ %205, %209 ]
  %216 = add nsw i32 %214, 1
  %217 = icmp eq i32 %216, %212
  br i1 %217, label %219, label %11, !llvm.loop !55

218:                                              ; preds = %3
  tail call void @free(ptr noundef nonnull %4) #16
  br label %225

219:                                              ; preds = %211
  %220 = load ptr, ptr %5, align 8, !tbaa !23
  tail call void @free(ptr noundef %4) #16
  %221 = icmp eq ptr %220, null
  br i1 %221, label %225, label %222

222:                                              ; preds = %219
  %223 = load i32, ptr %1, align 4, !tbaa !6
  %224 = sub nsw i32 1, %223
  store i32 %224, ptr %1, align 4, !tbaa !6
  br label %225

225:                                              ; preds = %218, %222, %219
  %226 = phi ptr [ null, %218 ], [ %220, %222 ], [ null, %219 ]
  ret ptr %226
}

; Function Attrs: nounwind uwtable
define dso_local ptr @make_play(i32 noundef %0) local_unnamed_addr #5 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4, !tbaa !6
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #16
  %4 = tail call noalias dereferenceable_or_null(32) ptr @malloc(i64 noundef 32) #15
  store ptr null, ptr @game_tree, align 8, !tbaa !34
  %5 = load i32, ptr @ncol, align 4, !tbaa !6
  %6 = sext i32 %5 to i64
  %7 = shl nsw i64 %6, 2
  %8 = tail call noalias ptr @malloc(i64 noundef %7) #15
  %9 = icmp eq i32 %5, 0
  br i1 %9, label %112, label %10

10:                                               ; preds = %1
  %11 = zext i32 %5 to i64
  %12 = shl nuw nsw i64 %11, 2
  tail call void @llvm.memset.p0.i64(ptr align 4 %8, i8 0, i64 %12, i1 false), !tbaa !6
  %13 = load i32, ptr %8, align 4, !tbaa !6
  %14 = add nsw i32 %13, -1
  store i32 %14, ptr %8, align 4, !tbaa !6
  br label %15

15:                                               ; preds = %10, %109
  %16 = phi ptr [ %4, %10 ], [ %111, %109 ]
  %17 = phi ptr [ %8, %10 ], [ %110, %109 ]
  br label %18

18:                                               ; preds = %15, %77
  %19 = phi ptr [ %74, %77 ], [ %16, %15 ]
  br label %20

20:                                               ; preds = %18, %26
  %21 = phi i64 [ 0, %18 ], [ %27, %26 ]
  %22 = getelementptr inbounds nuw i32, ptr %17, i64 %21
  %23 = load i32, ptr %22, align 4, !tbaa !6
  %24 = load i32, ptr @nrow, align 4, !tbaa !6
  %25 = icmp eq i32 %23, %24
  br i1 %25, label %26, label %31

26:                                               ; preds = %20
  %27 = add nuw nsw i64 %21, 1
  store i32 0, ptr %22, align 4, !tbaa !6
  %28 = load i32, ptr @ncol, align 4, !tbaa !6
  %29 = zext i32 %28 to i64
  %30 = icmp eq i64 %27, %29
  br i1 %30, label %112, label %20, !llvm.loop !10

31:                                               ; preds = %20
  %32 = add nsw i32 %23, 1
  store i32 %32, ptr %22, align 4, !tbaa !6
  %33 = load i32, ptr @ncol, align 4, !tbaa !6
  %34 = icmp eq i32 %33, 0
  br i1 %34, label %47, label %35

35:                                               ; preds = %31
  %36 = load i32, ptr @nrow, align 4, !tbaa !6
  %37 = zext i32 %33 to i64
  br label %41

38:                                               ; preds = %41
  %39 = add nuw nsw i64 %42, 1
  %40 = icmp eq i64 %39, %37
  br i1 %40, label %47, label %41, !llvm.loop !22

41:                                               ; preds = %38, %35
  %42 = phi i64 [ 0, %35 ], [ %39, %38 ]
  %43 = phi i32 [ %36, %35 ], [ %45, %38 ]
  %44 = getelementptr inbounds nuw i32, ptr %17, i64 %42
  %45 = load i32, ptr %44, align 4, !tbaa !6
  %46 = icmp sgt i32 %45, %43
  br i1 %46, label %109, label %38

47:                                               ; preds = %38, %31
  %48 = tail call noalias dereferenceable_or_null(32) ptr @malloc(i64 noundef 32) #15
  %49 = getelementptr inbounds nuw i8, ptr %19, i64 24
  store ptr %48, ptr %49, align 8, !tbaa !29
  %50 = load ptr, ptr @game_tree, align 8, !tbaa !34
  %51 = icmp eq ptr %50, null
  br i1 %51, label %52, label %53

52:                                               ; preds = %47
  store ptr %48, ptr @game_tree, align 8, !tbaa !34
  br label %53

53:                                               ; preds = %52, %47
  %54 = sext i32 %33 to i64
  %55 = shl nsw i64 %54, 2
  %56 = tail call noalias ptr @malloc(i64 noundef %55) #15
  br i1 %34, label %67, label %57

57:                                               ; preds = %53
  %58 = add i32 %33, -1
  %59 = sext i32 %58 to i64
  %60 = zext i32 %58 to i64
  %61 = sub nsw i64 %59, %60
  %62 = shl nsw i64 %61, 2
  %63 = getelementptr i8, ptr %56, i64 %62
  %64 = getelementptr i8, ptr %17, i64 %62
  %65 = zext i32 %33 to i64
  %66 = shl nuw nsw i64 %65, 2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %63, ptr readonly align 4 %64, i64 %66, i1 false), !tbaa !6
  br label %67

67:                                               ; preds = %53, %57
  %68 = getelementptr inbounds nuw i8, ptr %48, i64 8
  store ptr %56, ptr %68, align 8, !tbaa !33
  %69 = call ptr @make_list(ptr noundef nonnull %17, ptr noundef nonnull %3, ptr noundef nonnull %2)
  %70 = load ptr, ptr %49, align 8, !tbaa !29
  %71 = getelementptr inbounds nuw i8, ptr %70, i64 16
  store ptr %69, ptr %71, align 8, !tbaa !32
  %72 = load i32, ptr %3, align 4, !tbaa !6
  store i32 %72, ptr %70, align 8, !tbaa !36
  %73 = getelementptr inbounds nuw i8, ptr %70, i64 24
  store ptr null, ptr %73, align 8, !tbaa !29
  %74 = load ptr, ptr %49, align 8, !tbaa !29
  %75 = load i32, ptr %2, align 4, !tbaa !6
  %76 = icmp eq i32 %75, 2
  br i1 %76, label %80, label %77

77:                                               ; preds = %67
  %78 = load i32, ptr @ncol, align 4, !tbaa !6
  %79 = icmp eq i32 %78, 0
  br i1 %79, label %112, label %18, !llvm.loop !57

80:                                               ; preds = %67
  tail call void @free(ptr noundef nonnull %17) #16
  %81 = load i32, ptr @nrow, align 4, !tbaa !6
  %82 = load i32, ptr @ncol, align 4, !tbaa !6
  %83 = sext i32 %82 to i64
  %84 = shl nsw i64 %83, 2
  %85 = tail call noalias ptr @malloc(i64 noundef %84) #15
  %86 = icmp eq i32 %82, 0
  br i1 %86, label %112, label %87

87:                                               ; preds = %80
  %88 = zext i32 %82 to i64
  %89 = icmp ult i32 %82, 8
  br i1 %89, label %102, label %90

90:                                               ; preds = %87
  %91 = and i64 %88, 4294967288
  %92 = insertelement <4 x i32> poison, i32 %81, i64 0
  %93 = shufflevector <4 x i32> %92, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %94

94:                                               ; preds = %94, %90
  %95 = phi i64 [ 0, %90 ], [ %98, %94 ]
  %96 = getelementptr inbounds nuw i32, ptr %85, i64 %95
  %97 = getelementptr inbounds nuw i8, ptr %96, i64 16
  store <4 x i32> %93, ptr %96, align 4, !tbaa !6
  store <4 x i32> %93, ptr %97, align 4, !tbaa !6
  %98 = add nuw i64 %95, 8
  %99 = icmp eq i64 %98, %91
  br i1 %99, label %100, label %94, !llvm.loop !58

100:                                              ; preds = %94
  %101 = icmp eq i64 %91, %88
  br i1 %101, label %109, label %102

102:                                              ; preds = %87, %100
  %103 = phi i64 [ 0, %87 ], [ %91, %100 ]
  br label %104

104:                                              ; preds = %102, %104
  %105 = phi i64 [ %107, %104 ], [ %103, %102 ]
  %106 = getelementptr inbounds nuw i32, ptr %85, i64 %105
  store i32 %81, ptr %106, align 4, !tbaa !6
  %107 = add nuw nsw i64 %105, 1
  %108 = icmp eq i64 %107, %88
  br i1 %108, label %109, label %104, !llvm.loop !59

109:                                              ; preds = %104, %41, %100
  %110 = phi ptr [ %85, %100 ], [ %17, %41 ], [ %85, %104 ]
  %111 = phi ptr [ %74, %100 ], [ %19, %41 ], [ %74, %104 ]
  br label %15, !llvm.loop !57

112:                                              ; preds = %80, %77, %26, %1
  %113 = getelementptr inbounds nuw i8, ptr %4, i64 24
  %114 = load ptr, ptr %113, align 8, !tbaa !29
  tail call void @free(ptr noundef %4) #16
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #16
  ret ptr %114
}

; Function Attrs: nounwind uwtable
define dso_local void @make_wanted(ptr noundef readonly captures(none) %0) local_unnamed_addr #5 {
  %2 = tail call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #15
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr null, ptr %3, align 8, !tbaa !23
  %4 = load i32, ptr @nrow, align 4, !tbaa !6
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %144, label %6

6:                                                ; preds = %1
  %7 = load i32, ptr @ncol, align 4, !tbaa !6
  %8 = icmp eq i32 %7, 0
  %9 = sext i32 %7 to i64
  %10 = shl nsw i64 %9, 2
  %11 = zext i32 %7 to i64
  %12 = add i32 %7, -1
  %13 = sext i32 %12 to i64
  %14 = zext i32 %12 to i64
  %15 = sub nsw i64 %13, %14
  %16 = shl nsw i64 %15, 2
  %17 = shl nuw nsw i64 %11, 2
  %18 = add nsw i32 %4, -1
  br i1 %8, label %144, label %19

19:                                               ; preds = %6
  %20 = insertelement <4 x i32> poison, i32 %4, i64 0
  %21 = shufflevector <4 x i32> %20, <4 x i32> poison, <4 x i32> zeroinitializer
  %22 = icmp ult i32 %7, 8
  %23 = and i64 %9, -8
  %24 = and i64 %9, 7
  %25 = icmp eq i64 %23, %9
  br label %26

26:                                               ; preds = %19, %137
  %27 = phi ptr [ %138, %137 ], [ %2, %19 ]
  %28 = phi i32 [ %140, %137 ], [ 0, %19 ]
  %29 = insertelement <4 x i32> poison, i32 %28, i64 0
  %30 = shufflevector <4 x i32> %29, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %31

31:                                               ; preds = %26, %126
  %32 = phi i64 [ 0, %26 ], [ %136, %126 ]
  %33 = phi ptr [ %27, %26 ], [ %133, %126 ]
  %34 = phi i32 [ 0, %26 ], [ %134, %126 ]
  %35 = sub i64 %11, %32
  %36 = tail call noalias ptr @malloc(i64 noundef %10) #15
  %37 = icmp eq i32 %34, 0
  br i1 %37, label %53, label %38

38:                                               ; preds = %31
  %39 = zext nneg i32 %34 to i64
  %40 = icmp ult i64 %32, 8
  br i1 %40, label %51, label %41

41:                                               ; preds = %38
  %42 = and i64 %32, -8
  br label %43

43:                                               ; preds = %43, %41
  %44 = phi i64 [ 0, %41 ], [ %47, %43 ]
  %45 = getelementptr inbounds nuw i32, ptr %36, i64 %44
  %46 = getelementptr inbounds nuw i8, ptr %45, i64 16
  store <4 x i32> %21, ptr %45, align 4, !tbaa !6
  store <4 x i32> %21, ptr %46, align 4, !tbaa !6
  %47 = add nuw i64 %44, 8
  %48 = icmp eq i64 %47, %42
  br i1 %48, label %49, label %43, !llvm.loop !60

49:                                               ; preds = %43
  %50 = icmp eq i64 %32, %42
  br i1 %50, label %53, label %51

51:                                               ; preds = %38, %49
  %52 = phi i64 [ 0, %38 ], [ %42, %49 ]
  br label %70

53:                                               ; preds = %70, %49, %31
  %54 = phi i64 [ 0, %31 ], [ %39, %49 ], [ %39, %70 ]
  %55 = icmp ult i64 %35, 8
  br i1 %55, label %68, label %56

56:                                               ; preds = %53
  %57 = and i64 %35, -8
  %58 = add i64 %54, %57
  %59 = getelementptr i32, ptr %36, i64 %54
  br label %60

60:                                               ; preds = %60, %56
  %61 = phi i64 [ 0, %56 ], [ %64, %60 ]
  %62 = getelementptr i32, ptr %59, i64 %61
  %63 = getelementptr inbounds nuw i8, ptr %62, i64 16
  store <4 x i32> %30, ptr %62, align 4, !tbaa !6
  store <4 x i32> %30, ptr %63, align 4, !tbaa !6
  %64 = add nuw i64 %61, 8
  %65 = icmp eq i64 %64, %57
  br i1 %65, label %66, label %60, !llvm.loop !61

66:                                               ; preds = %60
  %67 = icmp eq i64 %35, %57
  br i1 %67, label %80, label %68

68:                                               ; preds = %53, %66
  %69 = phi i64 [ %54, %53 ], [ %58, %66 ]
  br label %75

70:                                               ; preds = %51, %70
  %71 = phi i64 [ %73, %70 ], [ %52, %51 ]
  %72 = getelementptr inbounds nuw i32, ptr %36, i64 %71
  store i32 %4, ptr %72, align 4, !tbaa !6
  %73 = add nuw nsw i64 %71, 1
  %74 = icmp eq i64 %73, %39
  br i1 %74, label %53, label %70, !llvm.loop !62

75:                                               ; preds = %68, %75
  %76 = phi i64 [ %78, %75 ], [ %69, %68 ]
  %77 = getelementptr inbounds nuw i32, ptr %36, i64 %76
  store i32 %28, ptr %77, align 4, !tbaa !6
  %78 = add nuw nsw i64 %76, 1
  %79 = icmp eq i64 %78, %11
  br i1 %79, label %80, label %75, !llvm.loop !63

80:                                               ; preds = %75, %66
  br i1 %22, label %100, label %81

81:                                               ; preds = %80, %81
  %82 = phi i64 [ %97, %81 ], [ 0, %80 ]
  %83 = xor i64 %82, -1
  %84 = add i64 %83, %9
  %85 = getelementptr inbounds i32, ptr %36, i64 %84
  %86 = getelementptr inbounds i8, ptr %85, i64 -12
  %87 = getelementptr inbounds i8, ptr %85, i64 -28
  %88 = load <4 x i32>, ptr %86, align 4, !tbaa !6
  %89 = load <4 x i32>, ptr %87, align 4, !tbaa !6
  %90 = getelementptr inbounds i32, ptr %0, i64 %84
  %91 = getelementptr inbounds i8, ptr %90, i64 -12
  %92 = getelementptr inbounds i8, ptr %90, i64 -28
  %93 = load <4 x i32>, ptr %91, align 4, !tbaa !6
  %94 = load <4 x i32>, ptr %92, align 4, !tbaa !6
  %95 = tail call <4 x i32> @llvm.smin.v4i32(<4 x i32> %88, <4 x i32> %93)
  %96 = tail call <4 x i32> @llvm.smin.v4i32(<4 x i32> %89, <4 x i32> %94)
  store <4 x i32> %95, ptr %86, align 4
  store <4 x i32> %96, ptr %87, align 4
  %97 = add nuw i64 %82, 8
  %98 = icmp eq i64 %97, %23
  br i1 %98, label %99, label %81, !llvm.loop !64

99:                                               ; preds = %81
  br i1 %25, label %111, label %100

100:                                              ; preds = %80, %99
  %101 = phi i64 [ %9, %80 ], [ %24, %99 ]
  br label %102

102:                                              ; preds = %100, %102
  %103 = phi i64 [ %104, %102 ], [ %101, %100 ]
  %104 = add nsw i64 %103, -1
  %105 = getelementptr inbounds i32, ptr %36, i64 %104
  %106 = load i32, ptr %105, align 4, !tbaa !6
  %107 = getelementptr inbounds i32, ptr %0, i64 %104
  %108 = load i32, ptr %107, align 4, !tbaa !6
  %109 = tail call i32 @llvm.smin.i32(i32 %106, i32 %108)
  store i32 %109, ptr %105, align 4
  %110 = icmp eq i64 %104, 0
  br i1 %110, label %111, label %102, !llvm.loop !65

111:                                              ; preds = %102, %99
  br label %112

112:                                              ; preds = %111, %115
  %113 = phi i64 [ %116, %115 ], [ %9, %111 ]
  %114 = icmp eq i64 %113, 0
  br i1 %114, label %124, label %115

115:                                              ; preds = %112
  %116 = add nsw i64 %113, -1
  %117 = getelementptr inbounds i32, ptr %36, i64 %116
  %118 = load i32, ptr %117, align 4, !tbaa !6
  %119 = getelementptr inbounds i32, ptr %0, i64 %116
  %120 = load i32, ptr %119, align 4, !tbaa !6
  %121 = icmp eq i32 %118, %120
  br i1 %121, label %112, label %122, !llvm.loop !21

122:                                              ; preds = %115
  %123 = icmp sgt i64 %113, 0
  br i1 %123, label %126, label %124

124:                                              ; preds = %122, %112
  %125 = select i1 %37, i32 %18, i32 %28
  tail call void @free(ptr noundef nonnull %36) #16
  br label %137

126:                                              ; preds = %122
  %127 = tail call noalias dereferenceable_or_null(16) ptr @malloc(i64 noundef 16) #15
  %128 = getelementptr inbounds nuw i8, ptr %33, i64 8
  store ptr %127, ptr %128, align 8, !tbaa !23
  %129 = tail call noalias ptr @malloc(i64 noundef %10) #15
  %130 = getelementptr i8, ptr %129, i64 %16
  %131 = getelementptr i8, ptr %36, i64 %16
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %130, ptr readonly align 4 %131, i64 %17, i1 false), !tbaa !6
  store ptr %129, ptr %127, align 8, !tbaa !28
  %132 = getelementptr inbounds nuw i8, ptr %127, i64 8
  store ptr null, ptr %132, align 8, !tbaa !23
  %133 = load ptr, ptr %128, align 8, !tbaa !23
  %134 = add nuw nsw i32 %34, 1
  tail call void @free(ptr noundef nonnull %36) #16
  %135 = icmp eq i32 %134, %7
  %136 = add i64 %32, 1
  br i1 %135, label %137, label %31, !llvm.loop !66

137:                                              ; preds = %126, %124
  %138 = phi ptr [ %33, %124 ], [ %133, %126 ]
  %139 = phi i32 [ %125, %124 ], [ %28, %126 ]
  %140 = add nsw i32 %139, 1
  %141 = icmp eq i32 %140, %4
  br i1 %141, label %142, label %26, !llvm.loop !67

142:                                              ; preds = %137
  %143 = load ptr, ptr %3, align 8, !tbaa !23
  br label %144

144:                                              ; preds = %6, %142, %1
  %145 = phi ptr [ %143, %142 ], [ null, %1 ], [ null, %6 ]
  tail call void @free(ptr noundef nonnull %2) #16
  store ptr %145, ptr @wanted, align 8, !tbaa !42
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noalias noundef ptr @get_good_move(ptr noundef readonly captures(address_is_null) %0) local_unnamed_addr #7 {
  %2 = icmp eq ptr %0, null
  br i1 %2, label %48, label %3

3:                                                ; preds = %1
  %4 = load i32, ptr @ncol, align 4
  %5 = sext i32 %4 to i64
  br label %6

6:                                                ; preds = %3, %31
  %7 = phi ptr [ %9, %31 ], [ %0, %3 ]
  %8 = getelementptr inbounds nuw i8, ptr %7, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !23
  %10 = icmp eq ptr %9, null
  %11 = load ptr, ptr %7, align 8, !tbaa !28
  br i1 %10, label %34, label %12

12:                                               ; preds = %6, %29
  %13 = phi ptr [ %30, %29 ], [ @game_tree, %6 ]
  %14 = load ptr, ptr %13, align 8, !tbaa !34
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 8
  %16 = load ptr, ptr %15, align 8, !tbaa !33
  br label %17

17:                                               ; preds = %20, %12
  %18 = phi i64 [ %21, %20 ], [ %5, %12 ]
  %19 = icmp eq i64 %18, 0
  br i1 %19, label %31, label %20

20:                                               ; preds = %17
  %21 = add nsw i64 %18, -1
  %22 = getelementptr inbounds i32, ptr %16, i64 %21
  %23 = load i32, ptr %22, align 4, !tbaa !6
  %24 = getelementptr inbounds i32, ptr %11, i64 %21
  %25 = load i32, ptr %24, align 4, !tbaa !6
  %26 = icmp eq i32 %23, %25
  br i1 %26, label %17, label %27, !llvm.loop !21

27:                                               ; preds = %20
  %28 = icmp sgt i64 %18, 0
  br i1 %28, label %29, label %31

29:                                               ; preds = %27
  %30 = getelementptr inbounds nuw i8, ptr %14, i64 24
  br label %12, !llvm.loop !35

31:                                               ; preds = %27, %17
  %32 = load i32, ptr %14, align 8, !tbaa !36
  %33 = icmp eq i32 %32, 0
  br i1 %33, label %34, label %6, !llvm.loop !68

34:                                               ; preds = %6, %31
  %35 = shl nsw i64 %5, 2
  %36 = tail call noalias ptr @malloc(i64 noundef %35) #15
  %37 = icmp eq i32 %4, 0
  br i1 %37, label %48, label %38

38:                                               ; preds = %34
  %39 = add i32 %4, -1
  %40 = sext i32 %39 to i64
  %41 = zext i32 %39 to i64
  %42 = sub nsw i64 %40, %41
  %43 = shl nsw i64 %42, 2
  %44 = getelementptr i8, ptr %36, i64 %43
  %45 = getelementptr i8, ptr %11, i64 %43
  %46 = zext i32 %4 to i64
  %47 = shl nuw nsw i64 %46, 2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %44, ptr readonly align 4 %45, i64 %47, i1 false), !tbaa !6
  br label %48

48:                                               ; preds = %38, %34, %1
  %49 = phi ptr [ null, %1 ], [ %36, %34 ], [ %36, %38 ]
  ret ptr %49
}

; Function Attrs: nofree nounwind uwtable
define dso_local noalias noundef ptr @get_winning_move(ptr noundef readonly captures(none) %0) local_unnamed_addr #7 {
  br label %2

2:                                                ; preds = %2, %1
  %3 = phi ptr [ %0, %1 ], [ %5, %2 ]
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %5 = load ptr, ptr %4, align 8, !tbaa !29
  %6 = icmp eq ptr %5, null
  br i1 %6, label %7, label %2, !llvm.loop !69

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %9 = load ptr, ptr %8, align 8, !tbaa !32
  %10 = icmp eq ptr %9, null
  br i1 %10, label %56, label %11

11:                                               ; preds = %7
  %12 = load i32, ptr @ncol, align 4
  %13 = sext i32 %12 to i64
  br label %14

14:                                               ; preds = %39, %11
  %15 = phi ptr [ %17, %39 ], [ %9, %11 ]
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 8
  %17 = load ptr, ptr %16, align 8, !tbaa !23
  %18 = icmp eq ptr %17, null
  %19 = load ptr, ptr %15, align 8, !tbaa !28
  br i1 %18, label %42, label %20

20:                                               ; preds = %14, %37
  %21 = phi ptr [ %38, %37 ], [ @game_tree, %14 ]
  %22 = load ptr, ptr %21, align 8, !tbaa !34
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 8
  %24 = load ptr, ptr %23, align 8, !tbaa !33
  br label %25

25:                                               ; preds = %28, %20
  %26 = phi i64 [ %29, %28 ], [ %13, %20 ]
  %27 = icmp eq i64 %26, 0
  br i1 %27, label %39, label %28

28:                                               ; preds = %25
  %29 = add nsw i64 %26, -1
  %30 = getelementptr inbounds i32, ptr %24, i64 %29
  %31 = load i32, ptr %30, align 4, !tbaa !6
  %32 = getelementptr inbounds i32, ptr %19, i64 %29
  %33 = load i32, ptr %32, align 4, !tbaa !6
  %34 = icmp eq i32 %31, %33
  br i1 %34, label %25, label %35, !llvm.loop !21

35:                                               ; preds = %28
  %36 = icmp sgt i64 %26, 0
  br i1 %36, label %37, label %39

37:                                               ; preds = %35
  %38 = getelementptr inbounds nuw i8, ptr %22, i64 24
  br label %20, !llvm.loop !35

39:                                               ; preds = %35, %25
  %40 = load i32, ptr %22, align 8, !tbaa !36
  %41 = icmp eq i32 %40, 0
  br i1 %41, label %42, label %14, !llvm.loop !68

42:                                               ; preds = %39, %14
  %43 = shl nsw i64 %13, 2
  %44 = tail call noalias ptr @malloc(i64 noundef %43) #15
  %45 = icmp eq i32 %12, 0
  br i1 %45, label %56, label %46

46:                                               ; preds = %42
  %47 = add i32 %12, -1
  %48 = sext i32 %47 to i64
  %49 = zext i32 %47 to i64
  %50 = sub nsw i64 %48, %49
  %51 = shl nsw i64 %50, 2
  %52 = getelementptr i8, ptr %44, i64 %51
  %53 = getelementptr i8, ptr %19, i64 %51
  %54 = zext i32 %12 to i64
  %55 = shl nuw nsw i64 %54, 2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %52, ptr readonly align 4 %53, i64 %55, i1 false), !tbaa !6
  br label %56

56:                                               ; preds = %7, %42, %46
  %57 = phi ptr [ null, %7 ], [ %44, %42 ], [ %44, %46 ]
  ret ptr %57
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable
define dso_local ptr @where(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #4 {
  %3 = load i32, ptr @ncol, align 4, !tbaa !6
  %4 = sext i32 %3 to i64
  br label %5

5:                                                ; preds = %21, %2
  %6 = phi ptr [ %1, %2 ], [ %23, %21 ]
  %7 = getelementptr inbounds nuw i8, ptr %6, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !33
  br label %9

9:                                                ; preds = %12, %5
  %10 = phi i64 [ %13, %12 ], [ %4, %5 ]
  %11 = icmp eq i64 %10, 0
  br i1 %11, label %24, label %12

12:                                               ; preds = %9
  %13 = add nsw i64 %10, -1
  %14 = getelementptr inbounds i32, ptr %8, i64 %13
  %15 = load i32, ptr %14, align 4, !tbaa !6
  %16 = getelementptr inbounds i32, ptr %0, i64 %13
  %17 = load i32, ptr %16, align 4, !tbaa !6
  %18 = icmp eq i32 %15, %17
  br i1 %18, label %9, label %19, !llvm.loop !21

19:                                               ; preds = %12
  %20 = icmp sgt i64 %10, 0
  br i1 %20, label %21, label %24

21:                                               ; preds = %19
  %22 = getelementptr inbounds nuw i8, ptr %6, i64 24
  %23 = load ptr, ptr %22, align 8, !tbaa !29
  br label %5, !llvm.loop !70

24:                                               ; preds = %19, %9
  %25 = getelementptr inbounds nuw i8, ptr %6, i64 16
  %26 = load ptr, ptr %25, align 8, !tbaa !32
  ret ptr %26
}

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @get_real_move(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, ptr noundef writeonly captures(none) %2, ptr noundef writeonly captures(none) initializes((0, 4)) %3) local_unnamed_addr #10 {
  br label %5

5:                                                ; preds = %5, %4
  %6 = phi i64 [ %13, %5 ], [ 0, %4 ]
  %7 = trunc nuw nsw i64 %6 to i32
  store i32 %7, ptr %3, align 4, !tbaa !6
  %8 = getelementptr inbounds nuw i32, ptr %0, i64 %6
  %9 = load i32, ptr %8, align 4, !tbaa !6
  %10 = getelementptr inbounds nuw i32, ptr %1, i64 %6
  %11 = load i32, ptr %10, align 4, !tbaa !6
  %12 = icmp eq i32 %9, %11
  %13 = add nuw nsw i64 %6, 1
  br i1 %12, label %5, label %14, !llvm.loop !71

14:                                               ; preds = %5
  store i32 %9, ptr %2, align 4, !tbaa !6
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.20)
  %2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.21)
  %3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.22)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13)
  store i32 7, ptr @ncol, align 4, !tbaa !6
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14)
  store i32 8, ptr @nrow, align 4, !tbaa !6
  %7 = tail call ptr @make_play(i32 noundef 1)
  %8 = load i32, ptr @nrow, align 4, !tbaa !6
  %9 = load i32, ptr @ncol, align 4, !tbaa !6
  %10 = sext i32 %9 to i64
  %11 = shl nsw i64 %10, 2
  %12 = tail call noalias ptr @malloc(i64 noundef %11) #15
  %13 = icmp eq i32 %9, 0
  br i1 %13, label %36, label %14

14:                                               ; preds = %0
  %15 = zext i32 %9 to i64
  %16 = icmp ult i32 %9, 8
  br i1 %16, label %29, label %17

17:                                               ; preds = %14
  %18 = and i64 %15, 4294967288
  %19 = insertelement <4 x i32> poison, i32 %8, i64 0
  %20 = shufflevector <4 x i32> %19, <4 x i32> poison, <4 x i32> zeroinitializer
  br label %21

21:                                               ; preds = %21, %17
  %22 = phi i64 [ 0, %17 ], [ %25, %21 ]
  %23 = getelementptr inbounds nuw i32, ptr %12, i64 %22
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  store <4 x i32> %20, ptr %23, align 4, !tbaa !6
  store <4 x i32> %20, ptr %24, align 4, !tbaa !6
  %25 = add nuw i64 %22, 8
  %26 = icmp eq i64 %25, %18
  br i1 %26, label %27, label %21, !llvm.loop !72

27:                                               ; preds = %21
  %28 = icmp eq i64 %18, %15
  br i1 %28, label %38, label %29

29:                                               ; preds = %14, %27
  %30 = phi i64 [ 0, %14 ], [ %18, %27 ]
  br label %31

31:                                               ; preds = %29, %31
  %32 = phi i64 [ %34, %31 ], [ %30, %29 ]
  %33 = getelementptr inbounds nuw i32, ptr %12, i64 %32
  store i32 %8, ptr %33, align 4, !tbaa !6
  %34 = add nuw nsw i64 %32, 1
  %35 = icmp eq i64 %34, %15
  br i1 %35, label %38, label %31, !llvm.loop !73

36:                                               ; preds = %0
  %37 = icmp eq ptr %12, null
  br i1 %37, label %126, label %38

38:                                               ; preds = %31, %27, %36
  br label %39

39:                                               ; preds = %38, %119
  %40 = phi i32 [ %123, %119 ], [ %9, %38 ]
  %41 = phi i32 [ %122, %119 ], [ 0, %38 ]
  %42 = phi ptr [ %97, %119 ], [ %12, %38 ]
  %43 = sext i32 %40 to i64
  br label %44

44:                                               ; preds = %60, %39
  %45 = phi ptr [ %7, %39 ], [ %62, %60 ]
  %46 = getelementptr inbounds nuw i8, ptr %45, i64 8
  %47 = load ptr, ptr %46, align 8, !tbaa !33
  br label %48

48:                                               ; preds = %51, %44
  %49 = phi i64 [ %52, %51 ], [ %43, %44 ]
  %50 = icmp eq i64 %49, 0
  br i1 %50, label %63, label %51

51:                                               ; preds = %48
  %52 = add nsw i64 %49, -1
  %53 = getelementptr inbounds i32, ptr %47, i64 %52
  %54 = load i32, ptr %53, align 4, !tbaa !6
  %55 = getelementptr inbounds i32, ptr %42, i64 %52
  %56 = load i32, ptr %55, align 4, !tbaa !6
  %57 = icmp eq i32 %54, %56
  br i1 %57, label %48, label %58, !llvm.loop !21

58:                                               ; preds = %51
  %59 = icmp sgt i64 %49, 0
  br i1 %59, label %60, label %63

60:                                               ; preds = %58
  %61 = getelementptr inbounds nuw i8, ptr %45, i64 24
  %62 = load ptr, ptr %61, align 8, !tbaa !29
  br label %44, !llvm.loop !70

63:                                               ; preds = %58, %48
  %64 = getelementptr inbounds nuw i8, ptr %45, i64 16
  %65 = load ptr, ptr %64, align 8, !tbaa !32
  %66 = icmp eq ptr %65, null
  br i1 %66, label %124, label %67

67:                                               ; preds = %63, %92
  %68 = phi ptr [ %70, %92 ], [ %65, %63 ]
  %69 = getelementptr inbounds nuw i8, ptr %68, i64 8
  %70 = load ptr, ptr %69, align 8, !tbaa !23
  %71 = icmp eq ptr %70, null
  %72 = load ptr, ptr %68, align 8, !tbaa !28
  br i1 %71, label %95, label %73

73:                                               ; preds = %67, %90
  %74 = phi ptr [ %91, %90 ], [ @game_tree, %67 ]
  %75 = load ptr, ptr %74, align 8, !tbaa !34
  %76 = getelementptr inbounds nuw i8, ptr %75, i64 8
  %77 = load ptr, ptr %76, align 8, !tbaa !33
  br label %78

78:                                               ; preds = %81, %73
  %79 = phi i64 [ %82, %81 ], [ %43, %73 ]
  %80 = icmp eq i64 %79, 0
  br i1 %80, label %92, label %81

81:                                               ; preds = %78
  %82 = add nsw i64 %79, -1
  %83 = getelementptr inbounds i32, ptr %77, i64 %82
  %84 = load i32, ptr %83, align 4, !tbaa !6
  %85 = getelementptr inbounds i32, ptr %72, i64 %82
  %86 = load i32, ptr %85, align 4, !tbaa !6
  %87 = icmp eq i32 %84, %86
  br i1 %87, label %78, label %88, !llvm.loop !21

88:                                               ; preds = %81
  %89 = icmp sgt i64 %79, 0
  br i1 %89, label %90, label %92

90:                                               ; preds = %88
  %91 = getelementptr inbounds nuw i8, ptr %75, i64 24
  br label %73, !llvm.loop !35

92:                                               ; preds = %88, %78
  %93 = load i32, ptr %75, align 8, !tbaa !36
  %94 = icmp eq i32 %93, 0
  br i1 %94, label %95, label %67, !llvm.loop !68

95:                                               ; preds = %92, %67
  %96 = shl nsw i64 %43, 2
  %97 = tail call noalias ptr @malloc(i64 noundef %96) #15
  %98 = icmp eq i32 %40, 0
  br i1 %98, label %109, label %99

99:                                               ; preds = %95
  %100 = add i32 %40, -1
  %101 = sext i32 %100 to i64
  %102 = zext i32 %100 to i64
  %103 = sub nsw i64 %101, %102
  %104 = shl nsw i64 %103, 2
  %105 = getelementptr i8, ptr %97, i64 %104
  %106 = getelementptr i8, ptr %72, i64 %104
  %107 = zext i32 %40 to i64
  %108 = shl nuw nsw i64 %107, 2
  tail call void @llvm.memcpy.p0.p0.i64(ptr align 4 %105, ptr readonly align 4 %106, i64 %108, i1 false), !tbaa !6
  br label %109

109:                                              ; preds = %95, %99
  %110 = icmp eq ptr %97, null
  br i1 %110, label %124, label %111

111:                                              ; preds = %109, %111
  %112 = phi i64 [ %118, %111 ], [ 0, %109 ]
  %113 = getelementptr inbounds nuw i32, ptr %97, i64 %112
  %114 = load i32, ptr %113, align 4, !tbaa !6
  %115 = getelementptr inbounds nuw i32, ptr %42, i64 %112
  %116 = load i32, ptr %115, align 4, !tbaa !6
  %117 = icmp eq i32 %114, %116
  %118 = add nuw nsw i64 %112, 1
  br i1 %117, label %111, label %119, !llvm.loop !71

119:                                              ; preds = %111
  %120 = trunc nuw nsw i64 %112 to i32
  %121 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15, i32 noundef %41, i32 noundef %114, i32 noundef %120)
  %122 = xor i32 %41, 1
  tail call void @free(ptr noundef nonnull %42) #16
  %123 = load i32, ptr @ncol, align 4, !tbaa !6
  br label %39, !llvm.loop !74

124:                                              ; preds = %109, %63
  %125 = xor i32 %41, 1
  br label %126

126:                                              ; preds = %124, %36
  %127 = phi i32 [ 1, %36 ], [ %125, %124 ]
  tail call void @dump_play(ptr noundef %7)
  %128 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.16, i32 noundef %127)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @putc(i32 noundef, ptr noundef captures(none)) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #11

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #13

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #14

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <4 x i32> @llvm.smin.v4i32(<4 x i32>, <4 x i32>) #13

attributes #0 = { mustprogress nofree nounwind willreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree nounwind memory(readwrite, argmem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nofree nounwind }
attributes #12 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #13 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #14 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #15 = { nounwind allocsize(0) }
attributes #16 = { nounwind }

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
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!13}
!13 = distinct !{!13, !14}
!14 = distinct !{!14, !"LVerDomain"}
!15 = !{!16}
!16 = distinct !{!16, !14}
!17 = distinct !{!17, !11, !18, !19}
!18 = !{!"llvm.loop.isvectorized", i32 1}
!19 = !{!"llvm.loop.unroll.runtime.disable"}
!20 = distinct !{!20, !11, !18}
!21 = distinct !{!21, !11}
!22 = distinct !{!22, !11}
!23 = !{!24, !27, i64 8}
!24 = !{!"_list", !25, i64 0, !27, i64 8}
!25 = !{!"p1 int", !26, i64 0}
!26 = !{!"any pointer", !8, i64 0}
!27 = !{!"p1 _ZTS5_list", !26, i64 0}
!28 = !{!24, !25, i64 0}
!29 = !{!30, !31, i64 24}
!30 = !{!"_play", !7, i64 0, !25, i64 8, !27, i64 16, !31, i64 24}
!31 = !{!"p1 _ZTS5_play", !26, i64 0}
!32 = !{!30, !27, i64 16}
!33 = !{!30, !25, i64 8}
!34 = !{!31, !31, i64 0}
!35 = distinct !{!35, !11}
!36 = !{!30, !7, i64 0}
!37 = !{!38, !38, i64 0}
!38 = !{!"p1 _ZTS8_IO_FILE", !26, i64 0}
!39 = distinct !{!39, !11}
!40 = distinct !{!40, !11}
!41 = distinct !{!41, !11}
!42 = !{!27, !27, i64 0}
!43 = distinct !{!43, !11}
!44 = distinct !{!44, !11, !18, !19}
!45 = distinct !{!45, !11, !18, !19}
!46 = distinct !{!46, !11, !19, !18}
!47 = distinct !{!47, !11, !19, !18}
!48 = distinct !{!48, !11, !18, !19}
!49 = distinct !{!49, !11, !18, !19}
!50 = distinct !{!50, !11, !19, !18}
!51 = distinct !{!51, !11, !19, !18}
!52 = distinct !{!52, !11, !18, !19}
!53 = distinct !{!53, !11, !19, !18}
!54 = distinct !{!54, !11}
!55 = distinct !{!55, !11, !56}
!56 = !{!"llvm.loop.unswitch.partial.disable"}
!57 = distinct !{!57, !11}
!58 = distinct !{!58, !11, !18, !19}
!59 = distinct !{!59, !11, !19, !18}
!60 = distinct !{!60, !11, !18, !19}
!61 = distinct !{!61, !11, !18, !19}
!62 = distinct !{!62, !11, !19, !18}
!63 = distinct !{!63, !11, !19, !18}
!64 = distinct !{!64, !11, !18, !19}
!65 = distinct !{!65, !11, !19, !18}
!66 = distinct !{!66, !11}
!67 = distinct !{!67, !11}
!68 = distinct !{!68, !11}
!69 = distinct !{!69, !11}
!70 = distinct !{!70, !11}
!71 = distinct !{!71, !11}
!72 = distinct !{!72, !11, !18, !19}
!73 = distinct !{!73, !11, !19, !18}
!74 = distinct !{!74, !11}
