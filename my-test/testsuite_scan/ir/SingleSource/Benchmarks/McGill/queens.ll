; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/McGill/queens.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/McGill/queens.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@printing = dso_local local_unnamed_addr global i32 1, align 4
@findall = dso_local local_unnamed_addr global i32 0, align 4
@solutions = dso_local local_unnamed_addr global i64 0, align 8
@progname = dso_local local_unnamed_addr global ptr null, align 8
@.str = private unnamed_addr constant [168 x i8] c"Usage:  %s [-ac] n\0A\09n\09Number of queens (rows and columns). An integer from 1 to 100.\0A\09-a\09Find and print all solutions.\0A\09-c\09Count all solutions, but do not print them.\0A\00", align 1
@queens = dso_local global i32 0, align 4
@stderr = external local_unnamed_addr global ptr, align 8
@.str.1 = private unnamed_addr constant [25 x i8] c"%s: Illegal option '%s'\0A\00", align 1
@.str.2 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.3 = private unnamed_addr constant [31 x i8] c"%s: Non-integer argument '%s'\0A\00", align 1
@.str.4 = private unnamed_addr constant [32 x i8] c"%s: n must be positive integer\0A\00", align 1
@.str.5 = private unnamed_addr constant [36 x i8] c"%s: Can't have more than %d queens\0A\00", align 1
@files = dso_local local_unnamed_addr global i32 0, align 4
@ranks = dso_local local_unnamed_addr global i32 0, align 4
@.str.7 = private unnamed_addr constant [32 x i8] c"%d queen%s on a %dx%d board...\0A\00", align 1
@.str.8 = private unnamed_addr constant [2 x i8] c"s\00", align 1
@.str.9 = private unnamed_addr constant [1 x i8] zeroinitializer, align 1
@stdout = external local_unnamed_addr global ptr, align 8
@file = dso_local local_unnamed_addr global [100 x i32] zeroinitializer, align 16
@bakdiag = dso_local local_unnamed_addr global [199 x i32] zeroinitializer, align 16
@fordiag = dso_local local_unnamed_addr global [199 x i32] zeroinitializer, align 16
@.str.11 = private unnamed_addr constant [28 x i8] c"...there are %ld solutions\0A\00", align 1
@.str.12 = private unnamed_addr constant [17 x i8] c"\0ASolution #%lu:\0A\00", align 1
@queen = dso_local local_unnamed_addr global [100 x i32] zeroinitializer, align 4
@str = private unnamed_addr constant [23 x i8] c"...there is 1 solution\00", align 4

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load ptr, ptr %1, align 8, !tbaa !6
  store ptr %3, ptr @progname, align 8, !tbaa !6
  store i32 0, ptr @printing, align 4, !tbaa !11
  store i32 14, ptr @queens, align 4, !tbaa !11
  store i32 1, ptr @findall, align 4, !tbaa !11
  %4 = icmp sgt i32 %0, 1
  br i1 %4, label %6, label %5

5:                                                ; preds = %2
  store i32 14, ptr @files, align 4, !tbaa !11
  store i32 14, ptr @ranks, align 4, !tbaa !11
  br label %55

6:                                                ; preds = %2
  %7 = zext nneg i32 %0 to i64
  br label %8

8:                                                ; preds = %6, %48
  %9 = phi i32 [ 14, %6 ], [ %49, %48 ]
  %10 = phi i64 [ 1, %6 ], [ %50, %48 ]
  %11 = getelementptr inbounds nuw ptr, ptr %1, i64 %10
  %12 = load ptr, ptr %11, align 8, !tbaa !6
  %13 = load i8, ptr %12, align 1, !tbaa !13
  %14 = icmp eq i8 %13, 45
  br i1 %14, label %15, label %28

15:                                               ; preds = %8, %20
  %16 = phi ptr [ %17, %20 ], [ %12, %8 ]
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 1
  %18 = load i8, ptr %17, align 1, !tbaa !13
  switch i8 %18, label %21 [
    i8 0, label %48
    i8 99, label %19
    i8 97, label %20
  ]

19:                                               ; preds = %15
  store i32 0, ptr @printing, align 4, !tbaa !11
  br label %20

20:                                               ; preds = %15, %19
  store i32 1, ptr @findall, align 4, !tbaa !11
  br label %15, !llvm.loop !14

21:                                               ; preds = %15
  %22 = load ptr, ptr @stderr, align 8, !tbaa !16
  %23 = load ptr, ptr @progname, align 8, !tbaa !6
  %24 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %22, ptr noundef nonnull @.str.1, ptr noundef %23, ptr noundef nonnull %12) #5
  %25 = load ptr, ptr @stderr, align 8, !tbaa !16
  %26 = load ptr, ptr @progname, align 8, !tbaa !6
  %27 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %25, ptr noundef nonnull @.str, ptr noundef %26) #5
  tail call void @exit(i32 noundef -1) #6
  unreachable

28:                                               ; preds = %8
  %29 = tail call i32 (ptr, ptr, ...) @__isoc99_sscanf(ptr noundef nonnull %12, ptr noundef nonnull @.str.2, ptr noundef nonnull @queens) #7
  %30 = icmp eq i32 %29, 1
  br i1 %30, label %35, label %31

31:                                               ; preds = %28
  %32 = load ptr, ptr @stderr, align 8, !tbaa !16
  %33 = load ptr, ptr @progname, align 8, !tbaa !6
  %34 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %32, ptr noundef nonnull @.str.3, ptr noundef %33, ptr noundef nonnull %12) #5
  tail call void @exit(i32 noundef -1) #6
  unreachable

35:                                               ; preds = %28
  %36 = load i32, ptr @queens, align 4, !tbaa !11
  %37 = icmp slt i32 %36, 1
  br i1 %37, label %38, label %42

38:                                               ; preds = %35
  %39 = load ptr, ptr @stderr, align 8, !tbaa !16
  %40 = load ptr, ptr @progname, align 8, !tbaa !6
  %41 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %39, ptr noundef nonnull @.str.4, ptr noundef %40) #5
  tail call void @exit(i32 noundef -1) #6
  unreachable

42:                                               ; preds = %35
  %43 = icmp samesign ugt i32 %36, 100
  br i1 %43, label %44, label %48

44:                                               ; preds = %42
  %45 = load ptr, ptr @stderr, align 8, !tbaa !16
  %46 = load ptr, ptr @progname, align 8, !tbaa !6
  %47 = tail call i32 (ptr, ptr, ...) @fprintf(ptr noundef %45, ptr noundef nonnull @.str.5, ptr noundef %46, i32 noundef 100) #5
  tail call void @exit(i32 noundef -1) #6
  unreachable

48:                                               ; preds = %15, %42
  %49 = phi i32 [ %36, %42 ], [ %9, %15 ]
  %50 = add nuw nsw i64 %10, 1
  %51 = icmp eq i64 %50, %7
  br i1 %51, label %52, label %8, !llvm.loop !18

52:                                               ; preds = %48
  store i32 %49, ptr @files, align 4, !tbaa !11
  store i32 %49, ptr @ranks, align 4, !tbaa !11
  %53 = icmp samesign ugt i32 %49, 1
  %54 = select i1 %53, ptr @.str.8, ptr @.str.9
  br label %55

55:                                               ; preds = %52, %5
  %56 = phi i32 [ 14, %5 ], [ %49, %52 ]
  %57 = phi ptr [ @.str.8, %5 ], [ %54, %52 ]
  %58 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef %56, ptr noundef nonnull %57, i32 noundef %56, i32 noundef %56)
  %59 = load ptr, ptr @stdout, align 8, !tbaa !16
  %60 = tail call i32 @fflush(ptr noundef %59)
  store i64 0, ptr @solutions, align 8, !tbaa !19
  store <4 x i32> splat (i32 101), ptr @file, align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 16), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 32), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 48), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 64), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 80), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 96), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 112), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 128), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 144), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 160), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 176), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 192), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 208), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 224), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 240), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 256), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 272), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 288), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 304), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 320), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 336), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 352), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @file, i64 368), align 16, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @file, i64 384), align 16, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @file, i64 388), align 4, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @file, i64 392), align 8, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @file, i64 396), align 4, !tbaa !11
  store <4 x i32> splat (i32 101), ptr @bakdiag, align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 16), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr @fordiag, align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 16), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 32), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 48), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 32), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 48), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 64), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 80), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 64), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 80), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 96), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 112), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 96), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 112), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 128), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 144), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 128), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 144), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 160), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 176), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 160), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 176), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 192), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 208), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 192), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 208), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 224), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 240), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 224), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 240), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 256), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 272), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 256), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 272), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 288), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 304), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 288), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 304), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 320), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 336), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 320), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 336), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 352), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 368), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 352), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 368), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 384), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 400), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 384), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 400), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 416), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 432), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 416), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 432), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 448), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 464), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 448), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 464), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 480), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 496), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 480), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 496), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 512), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 528), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 512), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 528), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 544), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 560), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 544), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 560), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 576), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 592), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 576), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 592), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 608), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 624), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 608), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 624), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 640), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 656), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 640), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 656), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 672), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 688), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 672), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 688), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 704), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 720), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 704), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 720), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 736), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 752), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 736), align 16, !tbaa !11
  store <4 x i32> splat (i32 101), ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 752), align 16, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 768), align 16, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 768), align 16, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 772), align 4, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 772), align 4, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 776), align 8, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 776), align 8, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 780), align 4, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 780), align 4, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 784), align 16, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 784), align 16, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 788), align 4, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 788), align 4, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @bakdiag, i64 792), align 8, !tbaa !11
  store i32 101, ptr getelementptr inbounds nuw (i8, ptr @fordiag, i64 792), align 8, !tbaa !11
  tail call void @find(i32 noundef 0)
  %61 = load i32, ptr @printing, align 4, !tbaa !11
  %62 = icmp ne i32 %61, 0
  %63 = load i64, ptr @solutions, align 8
  %64 = icmp ne i64 %63, 0
  %65 = select i1 %62, i1 %64, i1 false
  br i1 %65, label %66, label %70

66:                                               ; preds = %55
  %67 = load ptr, ptr @stdout, align 8, !tbaa !16
  %68 = tail call i32 @putc(i32 noundef 10, ptr noundef %67)
  %69 = load i64, ptr @solutions, align 8, !tbaa !19
  br label %70

70:                                               ; preds = %66, %55
  %71 = phi i64 [ %69, %66 ], [ %63, %55 ]
  %72 = icmp eq i64 %71, 1
  br i1 %72, label %73, label %75

73:                                               ; preds = %70
  %74 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %77

75:                                               ; preds = %70
  %76 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i64 noundef %71)
  br label %77

77:                                               ; preds = %75, %73
  tail call void @exit(i32 noundef 0) #8
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef i32 @fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #2

; Function Attrs: nofree nounwind
declare noundef i32 @__isoc99_sscanf(ptr noundef readonly captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @fflush(ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @find(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @queens, align 4, !tbaa !11
  %3 = icmp eq i32 %0, %2
  br i1 %3, label %4, label %14

4:                                                ; preds = %1
  %5 = load i64, ptr @solutions, align 8, !tbaa !19
  %6 = add i64 %5, 1
  store i64 %6, ptr @solutions, align 8, !tbaa !19
  %7 = load i32, ptr @printing, align 4, !tbaa !11
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %10, label %9

9:                                                ; preds = %4
  tail call void @pboard()
  br label %10

10:                                               ; preds = %9, %4
  %11 = load i32, ptr @findall, align 4, !tbaa !11
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %48

13:                                               ; preds = %10
  tail call void @exit(i32 noundef 0) #8
  unreachable

14:                                               ; preds = %1
  %15 = load i32, ptr @files, align 4, !tbaa !11
  %16 = icmp sgt i32 %15, 0
  br i1 %16, label %17, label %48

17:                                               ; preds = %14
  %18 = add nsw i32 %15, %0
  %19 = sext i32 %18 to i64
  %20 = getelementptr i32, ptr @bakdiag, i64 %19
  %21 = sext i32 %0 to i64
  %22 = getelementptr inbounds i32, ptr @fordiag, i64 %21
  %23 = getelementptr inbounds i32, ptr @queen, i64 %21
  %24 = add nsw i32 %0, 1
  br label %25

25:                                               ; preds = %17, %42
  %26 = phi i32 [ %15, %17 ], [ %43, %42 ]
  %27 = phi ptr [ %20, %17 ], [ %31, %42 ]
  %28 = phi ptr [ %22, %17 ], [ %46, %42 ]
  %29 = phi ptr [ @file, %17 ], [ %45, %42 ]
  %30 = phi i32 [ 0, %17 ], [ %44, %42 ]
  %31 = getelementptr i8, ptr %27, i64 -4
  %32 = load i32, ptr %29, align 4, !tbaa !11
  %33 = icmp slt i32 %32, %0
  br i1 %33, label %42, label %34

34:                                               ; preds = %25
  %35 = load i32, ptr %28, align 4, !tbaa !11
  %36 = icmp slt i32 %35, %0
  br i1 %36, label %42, label %37

37:                                               ; preds = %34
  %38 = load i32, ptr %31, align 4, !tbaa !11
  %39 = icmp slt i32 %38, %0
  br i1 %39, label %42, label %40

40:                                               ; preds = %37
  store i32 %30, ptr %23, align 4, !tbaa !11
  store i32 %0, ptr %31, align 4, !tbaa !11
  store i32 %0, ptr %28, align 4, !tbaa !11
  store i32 %0, ptr %29, align 4, !tbaa !11
  tail call void @find(i32 noundef %24)
  store i32 101, ptr %31, align 4, !tbaa !11
  store i32 101, ptr %28, align 4, !tbaa !11
  store i32 101, ptr %29, align 4, !tbaa !11
  %41 = load i32, ptr @files, align 4, !tbaa !11
  br label %42

42:                                               ; preds = %25, %34, %37, %40
  %43 = phi i32 [ %26, %25 ], [ %26, %34 ], [ %26, %37 ], [ %41, %40 ]
  %44 = add nuw nsw i32 %30, 1
  %45 = getelementptr inbounds nuw i8, ptr %29, i64 4
  %46 = getelementptr inbounds nuw i8, ptr %28, i64 4
  %47 = icmp slt i32 %44, %43
  br i1 %47, label %25, label %48, !llvm.loop !21

48:                                               ; preds = %42, %14, %10
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @pboard() local_unnamed_addr #3 {
  %1 = load i32, ptr @findall, align 4, !tbaa !11
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %6, label %3

3:                                                ; preds = %0
  %4 = load i64, ptr @solutions, align 8, !tbaa !19
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i64 noundef %4)
  br label %6

6:                                                ; preds = %3, %0
  %7 = load i32, ptr @ranks, align 4, !tbaa !11
  %8 = icmp sgt i32 %7, 0
  br i1 %8, label %9, label %34

9:                                                ; preds = %6, %27
  %10 = phi i64 [ %30, %27 ], [ 0, %6 ]
  %11 = load i32, ptr @files, align 4, !tbaa !11
  %12 = icmp sgt i32 %11, 0
  br i1 %12, label %13, label %27

13:                                               ; preds = %9
  %14 = getelementptr inbounds nuw i32, ptr @queen, i64 %10
  br label %15

15:                                               ; preds = %13, %15
  %16 = phi i32 [ 0, %13 ], [ %24, %15 ]
  %17 = load ptr, ptr @stdout, align 8, !tbaa !16
  %18 = tail call i32 @putc(i32 noundef 32, ptr noundef %17)
  %19 = load i32, ptr %14, align 4, !tbaa !11
  %20 = icmp eq i32 %16, %19
  %21 = load ptr, ptr @stdout, align 8, !tbaa !16
  %22 = select i1 %20, i32 81, i32 45
  %23 = tail call i32 @putc(i32 noundef %22, ptr noundef %21)
  %24 = add nuw nsw i32 %16, 1
  %25 = load i32, ptr @files, align 4, !tbaa !11
  %26 = icmp slt i32 %24, %25
  br i1 %26, label %15, label %27, !llvm.loop !22

27:                                               ; preds = %15, %9
  %28 = load ptr, ptr @stdout, align 8, !tbaa !16
  %29 = tail call i32 @putc(i32 noundef 10, ptr noundef %28)
  %30 = add nuw nsw i64 %10, 1
  %31 = load i32, ptr @ranks, align 4, !tbaa !11
  %32 = sext i32 %31 to i64
  %33 = icmp slt i64 %30, %32
  br i1 %33, label %9, label %34, !llvm.loop !23

34:                                               ; preds = %27, %6
  %35 = load ptr, ptr @stdout, align 8, !tbaa !16
  %36 = tail call i32 @fflush(ptr noundef %35)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @putc(i32 noundef, ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #4

attributes #0 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind }
attributes #5 = { cold nounwind }
attributes #6 = { cold noreturn nounwind }
attributes #7 = { nounwind }
attributes #8 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = !{!9, !9, i64 0}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.mustprogress"}
!16 = !{!17, !17, i64 0}
!17 = !{!"p1 _ZTS8_IO_FILE", !8, i64 0}
!18 = distinct !{!18, !15}
!19 = !{!20, !20, i64 0}
!20 = !{!"long", !9, i64 0}
!21 = distinct !{!21, !15}
!22 = distinct !{!22, !15}
!23 = distinct !{!23, !15}
