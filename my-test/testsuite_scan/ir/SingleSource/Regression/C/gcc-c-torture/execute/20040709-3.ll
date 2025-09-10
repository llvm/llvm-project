; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20040709-3.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20040709-3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.A = type { i16 }
%struct.B = type <{ i16, i32 }>
%struct.C = type <{ i32, i16 }>
%struct.D = type { i64 }
%struct.E = type { i64, i64 }
%struct.F = type { i64, i64 }
%struct.G = type <{ i8, i64 }>
%struct.H = type <{ i16, i64 }>
%struct.I = type <{ i8, i64 }>
%struct.J = type { i16, i16 }
%struct.K = type { i32 }
%struct.L = type { i32, i32 }
%struct.M = type { i32, i32 }
%struct.N = type { i64 }
%struct.O = type { i64, i64 }
%struct.P = type { i64, i64 }
%struct.Q = type <{ i16, i64 }>
%struct.R = type <{ i16, i64 }>
%struct.S = type <{ i16, i64 }>
%struct.T = type { i16, i16 }
%struct.U = type <{ i16, i64 }>
%struct.V = type { i16, i16 }
%struct.W = type <{ fp128, i32 }>
%struct.X = type <{ i32, fp128 }>
%struct.Y = type <{ i32, fp128 }>
%struct.Z = type <{ fp128, i32 }>

@myrnd.s = internal unnamed_addr global i32 1388815473, align 4
@sA = dso_local local_unnamed_addr global %struct.A zeroinitializer, align 4
@sB = dso_local local_unnamed_addr global %struct.B zeroinitializer, align 4
@sC = dso_local local_unnamed_addr global %struct.C zeroinitializer, align 4
@sD = dso_local local_unnamed_addr global %struct.D zeroinitializer, align 8
@sE = dso_local local_unnamed_addr global %struct.E zeroinitializer, align 4
@sF = dso_local local_unnamed_addr global %struct.F zeroinitializer, align 8
@sG = dso_local local_unnamed_addr global %struct.G zeroinitializer, align 8
@sH = dso_local local_unnamed_addr global %struct.H zeroinitializer, align 8
@sI = dso_local local_unnamed_addr global %struct.I zeroinitializer, align 8
@sJ = dso_local local_unnamed_addr global %struct.J zeroinitializer, align 4
@sK = dso_local local_unnamed_addr global %struct.K zeroinitializer, align 4
@sL = dso_local local_unnamed_addr global %struct.L zeroinitializer, align 4
@sM = dso_local local_unnamed_addr global %struct.M zeroinitializer, align 4
@sN = dso_local local_unnamed_addr global %struct.N zeroinitializer, align 8
@sO = dso_local local_unnamed_addr global %struct.O zeroinitializer, align 4
@sP = dso_local local_unnamed_addr global %struct.P zeroinitializer, align 8
@sQ = dso_local local_unnamed_addr global %struct.Q zeroinitializer, align 8
@sR = dso_local local_unnamed_addr global %struct.R zeroinitializer, align 8
@sS = dso_local local_unnamed_addr global %struct.S zeroinitializer, align 8
@sT = dso_local local_unnamed_addr global %struct.T zeroinitializer, align 4
@sU = dso_local local_unnamed_addr global %struct.U zeroinitializer, align 8
@sV = dso_local local_unnamed_addr global %struct.V zeroinitializer, align 4
@sW = dso_local local_unnamed_addr global %struct.W zeroinitializer, align 16
@sX = dso_local local_unnamed_addr global %struct.X zeroinitializer, align 4
@sY = dso_local local_unnamed_addr global %struct.Y zeroinitializer, align 4
@sZ = dso_local local_unnamed_addr global %struct.Z zeroinitializer, align 16

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @myrnd() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  store i32 %3, ptr @myrnd.s, align 4, !tbaa !6
  %4 = lshr i32 %3, 16
  %5 = and i32 %4, 2047
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i16 @retmeA(i64 %0) local_unnamed_addr #1 {
  %2 = trunc i64 %0 to i16
  ret i16 %2
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @fn1A(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sA, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 5
  %5 = add i16 %4, %3
  %6 = and i16 %5, 2047
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2A(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sA, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 5
  %5 = add i16 %4, %3
  %6 = and i16 %5, 2047
  %7 = urem i16 %6, 15
  %8 = zext nneg i16 %7 to i32
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @retitA() local_unnamed_addr #3 {
  %1 = load i16, ptr @sA, align 4
  %2 = lshr i16 %1, 5
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @fn3A(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sA, align 4
  %3 = trunc i32 %0 to i16
  %4 = shl i16 %3, 5
  %5 = add i16 %2, %4
  store i16 %5, ptr @sA, align 4
  %6 = lshr i16 %5, 5
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testA() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sA, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sA, i64 1), align 1, !tbaa !10
  %10 = load i16, ptr @sA, align 4
  %11 = and i16 %10, 31
  %12 = mul i32 %7, -2139243339
  %13 = add i32 %12, -1492899873
  %14 = lshr i32 %13, 16
  %15 = mul i32 %13, 1103515245
  %16 = add i32 %15, 12345
  store i32 %16, ptr @myrnd.s, align 4, !tbaa !6
  %17 = trunc nuw i32 %14 to i16
  %18 = shl i16 %17, 5
  %19 = or disjoint i16 %18, %11
  store i16 %19, ptr @sA, align 4
  %20 = lshr i32 %16, 16
  %21 = trunc nuw i32 %20 to i16
  %22 = add i16 %17, %21
  %23 = and i16 %22, 2047
  %24 = urem i16 %23, 15
  %25 = add nuw nsw i32 %20, %14
  %26 = trunc i32 %25 to i16
  %27 = and i16 %26, 2047
  %28 = urem i16 %27, 15
  %29 = icmp eq i16 %28, %24
  br i1 %29, label %31, label %30

30:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

31:                                               ; preds = %0
  %32 = mul i32 %16, 1103515245
  %33 = add i32 %32, 12345
  %34 = lshr i32 %33, 16
  %35 = mul i32 %33, 1103515245
  %36 = add i32 %35, 12345
  store i32 %36, ptr @myrnd.s, align 4, !tbaa !6
  %37 = lshr i32 %36, 16
  %38 = trunc nuw i32 %34 to i16
  %39 = shl i16 %38, 5
  %40 = or disjoint i16 %39, %11
  %41 = trunc nuw i32 %37 to i16
  %42 = shl i16 %41, 5
  %43 = add i16 %40, %42
  store i16 %43, ptr @sA, align 4
  %44 = lshr i16 %43, 5
  %45 = zext nneg i16 %44 to i32
  %46 = add nuw nsw i32 %37, %34
  %47 = and i32 %46, 2047
  %48 = icmp eq i32 %47, %45
  br i1 %48, label %50, label %49

49:                                               ; preds = %31
  tail call void @abort() #9
  unreachable

50:                                               ; preds = %31
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #5

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i48 @retmeB(i64 %0) local_unnamed_addr #1 {
  %2 = trunc i64 %0 to i48
  ret i48 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @fn1B(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sB, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 5
  %5 = add i16 %4, %3
  %6 = and i16 %5, 2047
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2B(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sB, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 5
  %5 = add i16 %4, %3
  %6 = and i16 %5, 2047
  %7 = urem i16 %6, 15
  %8 = zext nneg i16 %7 to i32
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @retitB() local_unnamed_addr #3 {
  %1 = load i16, ptr @sB, align 4
  %2 = lshr i16 %1, 5
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @fn3B(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sB, align 4
  %3 = trunc i32 %0 to i16
  %4 = shl i16 %3, 5
  %5 = add i16 %2, %4
  store i16 %5, ptr @sB, align 4
  %6 = lshr i16 %5, 5
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testB() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sB, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sB, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sB, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sB, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sB, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sB, i64 5), align 1, !tbaa !10
  %26 = load i16, ptr @sB, align 4
  %27 = and i16 %26, 31
  %28 = mul i32 %23, -2139243339
  %29 = add i32 %28, -1492899873
  %30 = lshr i32 %29, 16
  %31 = mul i32 %29, 1103515245
  %32 = add i32 %31, 12345
  store i32 %32, ptr @myrnd.s, align 4, !tbaa !6
  %33 = trunc nuw i32 %30 to i16
  %34 = shl i16 %33, 5
  %35 = or disjoint i16 %34, %27
  store i16 %35, ptr @sB, align 4
  %36 = lshr i32 %32, 16
  %37 = trunc nuw i32 %36 to i16
  %38 = add i16 %33, %37
  %39 = and i16 %38, 2047
  %40 = urem i16 %39, 15
  %41 = add nuw nsw i32 %36, %30
  %42 = trunc i32 %41 to i16
  %43 = and i16 %42, 2047
  %44 = urem i16 %43, 15
  %45 = icmp eq i16 %44, %40
  br i1 %45, label %47, label %46

46:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

47:                                               ; preds = %0
  %48 = mul i32 %32, 1103515245
  %49 = add i32 %48, 12345
  %50 = lshr i32 %49, 16
  %51 = mul i32 %49, 1103515245
  %52 = add i32 %51, 12345
  store i32 %52, ptr @myrnd.s, align 4, !tbaa !6
  %53 = lshr i32 %52, 16
  %54 = trunc nuw i32 %50 to i16
  %55 = shl i16 %54, 5
  %56 = or disjoint i16 %55, %27
  %57 = trunc nuw i32 %53 to i16
  %58 = shl i16 %57, 5
  %59 = add i16 %56, %58
  store i16 %59, ptr @sB, align 4
  %60 = lshr i16 %59, 5
  %61 = zext nneg i16 %60 to i32
  %62 = add nuw nsw i32 %53, %50
  %63 = and i32 %62, 2047
  %64 = icmp eq i32 %63, %61
  br i1 %64, label %66, label %65

65:                                               ; preds = %47
  tail call void @abort() #9
  unreachable

66:                                               ; preds = %47
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i48 @retmeC(i64 %0) local_unnamed_addr #1 {
  %2 = trunc i64 %0 to i48
  ret i48 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @fn1C(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 1, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = shl i16 %3, 5
  %5 = add i16 %2, %4
  %6 = zext i16 %5 to i64
  %7 = shl nuw nsw i64 %6, 32
  %8 = trunc nuw i64 %7 to i48
  %9 = lshr i48 %8, 37
  %10 = trunc nuw nsw i48 %9 to i32
  ret i32 %10
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2C(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 1, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 5
  %5 = add i16 %4, %3
  %6 = and i16 %5, 2047
  %7 = urem i16 %6, 15
  %8 = zext nneg i16 %7 to i32
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @retitC() local_unnamed_addr #3 {
  %1 = load i16, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 1
  %2 = lshr i16 %1, 5
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2048) i32 @fn3C(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 1
  %3 = trunc i32 %0 to i16
  %4 = shl i16 %3, 5
  %5 = add i16 %2, %4
  store i16 %5, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 1
  %6 = lshr i16 %5, 5
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testC() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sC, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 5), align 1, !tbaa !10
  %26 = load i16, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 4
  %27 = mul i32 %23, 1103515245
  %28 = add i32 %27, 12345
  %29 = lshr i32 %28, 16
  %30 = mul i32 %28, 1103515245
  %31 = add i32 %30, 12345
  store i32 %31, ptr @myrnd.s, align 4, !tbaa !6
  %32 = trunc nuw i32 %29 to i16
  %33 = shl i16 %32, 5
  %34 = and i16 %26, 31
  %35 = or disjoint i16 %33, %34
  store i16 %35, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 4
  %36 = lshr i32 %31, 16
  %37 = trunc nuw i32 %36 to i16
  %38 = shl i16 %37, 5
  %39 = add i16 %38, %35
  %40 = zext i16 %39 to i64
  %41 = shl nuw nsw i64 %40, 32
  %42 = trunc nuw i64 %41 to i48
  %43 = lshr i48 %42, 37
  %44 = trunc nuw nsw i48 %43 to i32
  %45 = add nuw nsw i32 %36, %29
  %46 = and i32 %45, 2047
  %47 = icmp eq i32 %46, %44
  br i1 %47, label %49, label %48

48:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

49:                                               ; preds = %0
  %50 = mul i32 %31, 1103515245
  %51 = add i32 %50, 12345
  %52 = lshr i32 %51, 16
  %53 = mul i32 %51, 1103515245
  %54 = add i32 %53, 12345
  store i32 %54, ptr @myrnd.s, align 4, !tbaa !6
  %55 = trunc nuw i32 %52 to i16
  %56 = shl i16 %55, 5
  %57 = or disjoint i16 %56, %34
  store i16 %57, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 4
  %58 = lshr i32 %54, 16
  %59 = trunc nuw i32 %58 to i16
  %60 = add i16 %55, %59
  %61 = and i16 %60, 2047
  %62 = urem i16 %61, 15
  %63 = add nuw nsw i32 %58, %52
  %64 = trunc i32 %63 to i16
  %65 = and i16 %64, 2047
  %66 = urem i16 %65, 15
  %67 = icmp eq i16 %66, %62
  br i1 %67, label %69, label %68

68:                                               ; preds = %49
  tail call void @abort() #9
  unreachable

69:                                               ; preds = %49
  %70 = mul i32 %54, 1103515245
  %71 = add i32 %70, 12345
  %72 = lshr i32 %71, 16
  %73 = mul i32 %71, 1103515245
  %74 = add i32 %73, 12345
  store i32 %74, ptr @myrnd.s, align 4, !tbaa !6
  %75 = lshr i32 %74, 16
  %76 = trunc nuw i32 %72 to i16
  %77 = shl i16 %76, 5
  %78 = or disjoint i16 %77, %34
  %79 = trunc nuw i32 %75 to i16
  %80 = shl i16 %79, 5
  %81 = add i16 %78, %80
  store i16 %81, ptr getelementptr inbounds nuw (i8, ptr @sC, i64 4), align 4
  %82 = lshr i16 %81, 5
  %83 = zext nneg i16 %82 to i32
  %84 = add nuw nsw i32 %75, %72
  %85 = and i32 %84, 2047
  %86 = icmp eq i32 %85, %83
  br i1 %86, label %88, label %87

87:                                               ; preds = %69
  tail call void @abort() #9
  unreachable

88:                                               ; preds = %69
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i64 @retmeD(i64 returned %0) local_unnamed_addr #1 {
  ret i64 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 536870912) i32 @fn1D(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sD, align 8, !tbaa !10
  %3 = lshr i64 %2, 35
  %4 = trunc nuw nsw i64 %3 to i32
  %5 = add i32 %0, %4
  %6 = and i32 %5, 536870911
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2D(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sD, align 8, !tbaa !10
  %3 = lshr i64 %2, 35
  %4 = trunc nuw nsw i64 %3 to i32
  %5 = add i32 %0, %4
  %6 = and i32 %5, 536870911
  %7 = urem i32 %6, 15
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 536870912) i32 @retitD() local_unnamed_addr #3 {
  %1 = load i64, ptr @sD, align 8
  %2 = lshr i64 %1, 35
  %3 = trunc nuw nsw i64 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 536870912) i32 @fn3D(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i64, ptr @sD, align 8
  %3 = lshr i64 %2, 35
  %4 = trunc nuw nsw i64 %3 to i32
  %5 = add i32 %0, %4
  %6 = and i32 %5, 536870911
  %7 = zext nneg i32 %6 to i64
  %8 = shl nuw i64 %7, 35
  %9 = and i64 %2, 34359738367
  %10 = or disjoint i64 %8, %9
  store i64 %10, ptr @sD, align 8
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @testD() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sD, align 8, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sD, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sD, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sD, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sD, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sD, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sD, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sD, i64 7), align 1, !tbaa !10
  %34 = load i64, ptr @sD, align 8
  %35 = and i64 %34, 34359738367
  %36 = mul i32 %31, -341751747
  %37 = add i32 %36, 229283573
  %38 = lshr i32 %37, 16
  %39 = and i32 %38, 2047
  %40 = mul i32 %37, 1103515245
  %41 = add i32 %40, 12345
  store i32 %41, ptr @myrnd.s, align 4, !tbaa !6
  %42 = lshr i32 %41, 16
  %43 = and i32 %42, 2047
  %44 = add nuw nsw i32 %43, %39
  %45 = zext nneg i32 %44 to i64
  %46 = shl nuw nsw i64 %45, 35
  %47 = or disjoint i64 %46, %35
  store i64 %47, ptr @sD, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeE([2 x i64] returned %0) local_unnamed_addr #1 {
  ret [2 x i64] %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 536870912) i32 @fn1E(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 8), align 1, !tbaa !10
  %3 = lshr i64 %2, 35
  %4 = trunc nuw nsw i64 %3 to i32
  %5 = add i32 %0, %4
  %6 = and i32 %5, 536870911
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2E(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 8), align 1, !tbaa !10
  %3 = lshr i64 %2, 35
  %4 = trunc nuw nsw i64 %3 to i32
  %5 = add i32 %0, %4
  %6 = and i32 %5, 536870911
  %7 = urem i32 %6, 15
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 536870912) i32 @retitE() local_unnamed_addr #3 {
  %1 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 8), align 1
  %2 = lshr i64 %1, 35
  %3 = trunc nuw nsw i64 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 536870912) i32 @fn3E(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 8), align 1
  %3 = lshr i64 %2, 35
  %4 = trunc nuw nsw i64 %3 to i32
  %5 = add i32 %0, %4
  %6 = and i32 %5, 536870911
  %7 = zext nneg i32 %6 to i64
  %8 = shl nuw i64 %7, 35
  %9 = and i64 %2, 34359738367
  %10 = or disjoint i64 %8, %9
  store i64 %10, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 8), align 1
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @testE() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sE, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 8), align 4, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = trunc i32 %40 to i8
  store i8 %41, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 9), align 1, !tbaa !10
  %42 = mul i32 %39, 1103515245
  %43 = add i32 %42, 12345
  %44 = lshr i32 %43, 16
  %45 = trunc i32 %44 to i8
  store i8 %45, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 10), align 2, !tbaa !10
  %46 = mul i32 %43, 1103515245
  %47 = add i32 %46, 12345
  %48 = lshr i32 %47, 16
  %49 = trunc i32 %48 to i8
  store i8 %49, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 11), align 1, !tbaa !10
  %50 = mul i32 %47, 1103515245
  %51 = add i32 %50, 12345
  %52 = lshr i32 %51, 16
  %53 = trunc i32 %52 to i8
  store i8 %53, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 12), align 4, !tbaa !10
  %54 = mul i32 %51, 1103515245
  %55 = add i32 %54, 12345
  %56 = lshr i32 %55, 16
  %57 = trunc i32 %56 to i8
  store i8 %57, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 13), align 1, !tbaa !10
  %58 = mul i32 %55, 1103515245
  %59 = add i32 %58, 12345
  %60 = lshr i32 %59, 16
  %61 = trunc i32 %60 to i8
  store i8 %61, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 14), align 2, !tbaa !10
  %62 = mul i32 %59, 1103515245
  %63 = add i32 %62, 12345
  %64 = lshr i32 %63, 16
  %65 = trunc i32 %64 to i8
  store i8 %65, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 15), align 1, !tbaa !10
  %66 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 8), align 4
  %67 = and i64 %66, 34359738367
  %68 = mul i32 %63, -341751747
  %69 = add i32 %68, 229283573
  %70 = lshr i32 %69, 16
  %71 = and i32 %70, 2047
  %72 = mul i32 %69, 1103515245
  %73 = add i32 %72, 12345
  store i32 %73, ptr @myrnd.s, align 4, !tbaa !6
  %74 = lshr i32 %73, 16
  %75 = and i32 %74, 2047
  %76 = add nuw nsw i32 %75, %71
  %77 = zext nneg i32 %76 to i64
  %78 = shl nuw nsw i64 %77, 35
  %79 = or disjoint i64 %78, %67
  store i64 %79, ptr getelementptr inbounds nuw (i8, ptr @sE, i64 8), align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeF([2 x i64] returned %0) local_unnamed_addr #1 {
  ret [2 x i64] %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 536870912) i32 @fn1F(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sF, align 8, !tbaa !10
  %3 = lshr i64 %2, 35
  %4 = trunc nuw nsw i64 %3 to i32
  %5 = add i32 %0, %4
  %6 = and i32 %5, 536870911
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2F(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sF, align 8, !tbaa !10
  %3 = lshr i64 %2, 35
  %4 = trunc nuw nsw i64 %3 to i32
  %5 = add i32 %0, %4
  %6 = and i32 %5, 536870911
  %7 = urem i32 %6, 15
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 536870912) i32 @retitF() local_unnamed_addr #3 {
  %1 = load i64, ptr @sF, align 8
  %2 = lshr i64 %1, 35
  %3 = trunc nuw nsw i64 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 536870912) i32 @fn3F(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i64, ptr @sF, align 8
  %3 = lshr i64 %2, 35
  %4 = trunc nuw nsw i64 %3 to i32
  %5 = add i32 %0, %4
  %6 = and i32 %5, 536870911
  %7 = zext nneg i32 %6 to i64
  %8 = shl nuw i64 %7, 35
  %9 = and i64 %2, 34359738367
  %10 = or disjoint i64 %8, %9
  store i64 %10, ptr @sF, align 8
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @testF() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sF, align 8, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 8), align 8, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = trunc i32 %40 to i8
  store i8 %41, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 9), align 1, !tbaa !10
  %42 = mul i32 %39, 1103515245
  %43 = add i32 %42, 12345
  %44 = lshr i32 %43, 16
  %45 = trunc i32 %44 to i8
  store i8 %45, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 10), align 2, !tbaa !10
  %46 = mul i32 %43, 1103515245
  %47 = add i32 %46, 12345
  %48 = lshr i32 %47, 16
  %49 = trunc i32 %48 to i8
  store i8 %49, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 11), align 1, !tbaa !10
  %50 = mul i32 %47, 1103515245
  %51 = add i32 %50, 12345
  %52 = lshr i32 %51, 16
  %53 = trunc i32 %52 to i8
  store i8 %53, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 12), align 4, !tbaa !10
  %54 = mul i32 %51, 1103515245
  %55 = add i32 %54, 12345
  %56 = lshr i32 %55, 16
  %57 = trunc i32 %56 to i8
  store i8 %57, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 13), align 1, !tbaa !10
  %58 = mul i32 %55, 1103515245
  %59 = add i32 %58, 12345
  %60 = lshr i32 %59, 16
  %61 = trunc i32 %60 to i8
  store i8 %61, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 14), align 2, !tbaa !10
  %62 = mul i32 %59, 1103515245
  %63 = add i32 %62, 12345
  %64 = lshr i32 %63, 16
  %65 = trunc i32 %64 to i8
  store i8 %65, ptr getelementptr inbounds nuw (i8, ptr @sF, i64 15), align 1, !tbaa !10
  %66 = load i64, ptr @sF, align 8
  %67 = and i64 %66, 34359738367
  %68 = mul i32 %63, -341751747
  %69 = add i32 %68, 229283573
  %70 = lshr i32 %69, 16
  %71 = and i32 %70, 2047
  %72 = mul i32 %69, 1103515245
  %73 = add i32 %72, 12345
  store i32 %73, ptr @myrnd.s, align 4, !tbaa !6
  %74 = lshr i32 %73, 16
  %75 = and i32 %74, 2047
  %76 = add nuw nsw i32 %75, %71
  %77 = zext nneg i32 %76 to i64
  %78 = shl nuw nsw i64 %77, 35
  %79 = or disjoint i64 %78, %67
  store i64 %79, ptr @sF, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeG([2 x i64] %0) local_unnamed_addr #1 {
  %2 = extractvalue [2 x i64] %0, 1
  %3 = and i64 %2, 255
  %4 = insertvalue [2 x i64] %0, i64 %3, 1
  ret [2 x i64] %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn1G(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sG, align 8
  %3 = zext i32 %0 to i64
  %4 = shl nuw nsw i64 %3, 2
  %5 = add i64 %2, %4
  %6 = trunc i64 %5 to i32
  %7 = lshr i32 %6, 2
  %8 = and i32 %7, 63
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2G(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i8, ptr @sG, align 8, !tbaa !10
  %3 = trunc i32 %0 to i8
  %4 = lshr i8 %2, 2
  %5 = add i8 %4, %3
  %6 = and i8 %5, 63
  %7 = urem i8 %6, 15
  %8 = zext nneg i8 %7 to i32
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @retitG() local_unnamed_addr #3 {
  %1 = load i8, ptr @sG, align 8
  %2 = lshr i8 %1, 2
  %3 = zext nneg i8 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn3G(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i8, ptr @sG, align 8
  %3 = trunc i32 %0 to i8
  %4 = shl i8 %3, 2
  %5 = add i8 %2, %4
  store i8 %5, ptr @sG, align 8
  %6 = lshr i8 %5, 2
  %7 = zext nneg i8 %6 to i32
  ret i32 %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testG() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sG, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sG, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sG, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sG, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sG, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sG, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sG, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sG, i64 8), align 8, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = mul i32 %39, 1103515245
  %42 = add i32 %41, 12345
  store i32 %42, ptr @myrnd.s, align 4, !tbaa !6
  %43 = lshr i32 %42, 16
  %44 = trunc i32 %40 to i8
  %45 = shl i8 %44, 2
  %46 = and i8 %5, 3
  %47 = or disjoint i8 %45, %46
  store i8 %47, ptr @sG, align 8
  %48 = load i64, ptr @sG, align 8
  %49 = trunc i64 %48 to i32
  %50 = lshr i32 %49, 2
  %51 = add nuw nsw i32 %50, %43
  %52 = add nuw nsw i32 %43, %40
  %53 = xor i32 %52, %51
  %54 = and i32 %53, 63
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %57, label %56

56:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

57:                                               ; preds = %0
  %58 = mul i32 %42, 1103515245
  %59 = add i32 %58, 12345
  %60 = lshr i32 %59, 16
  %61 = mul i32 %59, 1103515245
  %62 = add i32 %61, 12345
  store i32 %62, ptr @myrnd.s, align 4, !tbaa !6
  %63 = lshr i32 %62, 16
  %64 = trunc i32 %60 to i8
  %65 = shl i8 %64, 2
  %66 = or disjoint i8 %65, %46
  store i8 %66, ptr @sG, align 8
  %67 = trunc i32 %63 to i8
  %68 = add i8 %67, %64
  %69 = and i8 %68, 63
  %70 = urem i8 %69, 15
  %71 = add nuw nsw i32 %63, %60
  %72 = trunc i32 %71 to i8
  %73 = and i8 %72, 63
  %74 = urem i8 %73, 15
  %75 = icmp eq i8 %74, %70
  br i1 %75, label %77, label %76

76:                                               ; preds = %57
  tail call void @abort() #9
  unreachable

77:                                               ; preds = %57
  %78 = mul i32 %62, 1103515245
  %79 = add i32 %78, 12345
  %80 = lshr i32 %79, 16
  %81 = mul i32 %79, 1103515245
  %82 = add i32 %81, 12345
  store i32 %82, ptr @myrnd.s, align 4, !tbaa !6
  %83 = lshr i32 %82, 16
  %84 = trunc i32 %80 to i8
  %85 = shl i8 %84, 2
  %86 = or disjoint i8 %85, %46
  %87 = trunc i32 %83 to i8
  %88 = shl i8 %87, 2
  %89 = add i8 %86, %88
  store i8 %89, ptr @sG, align 8
  %90 = lshr i8 %89, 2
  %91 = zext nneg i8 %90 to i32
  %92 = add nuw nsw i32 %83, %80
  %93 = and i32 %92, 63
  %94 = icmp eq i32 %93, %91
  br i1 %94, label %96, label %95

95:                                               ; preds = %77
  tail call void @abort() #9
  unreachable

96:                                               ; preds = %77
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeH([2 x i64] %0) local_unnamed_addr #1 {
  %2 = extractvalue [2 x i64] %0, 1
  %3 = and i64 %2, 65535
  %4 = insertvalue [2 x i64] %0, i64 %3, 1
  ret [2 x i64] %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 256) i32 @fn1H(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sH, align 8
  %3 = zext i32 %0 to i64
  %4 = shl nuw nsw i64 %3, 8
  %5 = add i64 %2, %4
  %6 = trunc i64 %5 to i32
  %7 = lshr i32 %6, 8
  %8 = and i32 %7, 255
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2H(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sH, align 8, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 8
  %5 = add i16 %4, %3
  %6 = trunc i16 %5 to i8
  %7 = urem i8 %6, 15
  %8 = zext nneg i8 %7 to i32
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 256) i32 @retitH() local_unnamed_addr #3 {
  %1 = load i16, ptr @sH, align 8
  %2 = lshr i16 %1, 8
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 256) i32 @fn3H(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sH, align 8
  %3 = trunc i32 %0 to i16
  %4 = shl i16 %3, 8
  %5 = add i16 %2, %4
  store i16 %5, ptr @sH, align 8
  %6 = lshr i16 %5, 8
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testH() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sH, align 8, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sH, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sH, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sH, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sH, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sH, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sH, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sH, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sH, i64 8), align 8, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = trunc i32 %40 to i8
  store i8 %41, ptr getelementptr inbounds nuw (i8, ptr @sH, i64 9), align 1, !tbaa !10
  %42 = load i16, ptr @sH, align 8
  %43 = mul i32 %39, 1103515245
  %44 = add i32 %43, 12345
  %45 = lshr i32 %44, 16
  %46 = mul i32 %44, 1103515245
  %47 = add i32 %46, 12345
  store i32 %47, ptr @myrnd.s, align 4, !tbaa !6
  %48 = lshr i32 %47, 16
  %49 = trunc nuw i32 %45 to i16
  %50 = shl i16 %49, 8
  %51 = and i16 %42, 255
  %52 = or disjoint i16 %50, %51
  store i16 %52, ptr @sH, align 8
  %53 = load i64, ptr @sH, align 8
  %54 = trunc i64 %53 to i32
  %55 = lshr i32 %54, 8
  %56 = add nuw nsw i32 %55, %48
  %57 = add nuw nsw i32 %48, %45
  %58 = xor i32 %57, %56
  %59 = and i32 %58, 255
  %60 = icmp eq i32 %59, 0
  br i1 %60, label %62, label %61

61:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

62:                                               ; preds = %0
  %63 = mul i32 %47, 1103515245
  %64 = add i32 %63, 12345
  %65 = lshr i32 %64, 16
  %66 = mul i32 %64, 1103515245
  %67 = add i32 %66, 12345
  store i32 %67, ptr @myrnd.s, align 4, !tbaa !6
  %68 = lshr i32 %67, 16
  %69 = trunc nuw i32 %65 to i16
  %70 = shl i16 %69, 8
  %71 = or disjoint i16 %70, %51
  store i16 %71, ptr @sH, align 8
  %72 = trunc nuw i32 %68 to i16
  %73 = add i16 %72, %69
  %74 = trunc i16 %73 to i8
  %75 = urem i8 %74, 15
  %76 = add nuw nsw i32 %68, %65
  %77 = trunc i32 %76 to i8
  %78 = urem i8 %77, 15
  %79 = icmp eq i8 %78, %75
  br i1 %79, label %81, label %80

80:                                               ; preds = %62
  tail call void @abort() #9
  unreachable

81:                                               ; preds = %62
  %82 = mul i32 %67, 1103515245
  %83 = add i32 %82, 12345
  %84 = lshr i32 %83, 16
  %85 = mul i32 %83, 1103515245
  %86 = add i32 %85, 12345
  store i32 %86, ptr @myrnd.s, align 4, !tbaa !6
  %87 = lshr i32 %86, 16
  %88 = trunc nuw i32 %84 to i16
  %89 = shl i16 %88, 8
  %90 = or disjoint i16 %89, %51
  %91 = trunc nuw i32 %87 to i16
  %92 = shl i16 %91, 8
  %93 = add i16 %90, %92
  store i16 %93, ptr @sH, align 8
  %94 = lshr i16 %93, 8
  %95 = zext nneg i16 %94 to i32
  %96 = add nuw nsw i32 %87, %84
  %97 = and i32 %96, 255
  %98 = icmp eq i32 %97, %95
  br i1 %98, label %100, label %99

99:                                               ; preds = %81
  tail call void @abort() #9
  unreachable

100:                                              ; preds = %81
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeI([2 x i64] %0) local_unnamed_addr #1 {
  %2 = extractvalue [2 x i64] %0, 1
  %3 = and i64 %2, 255
  %4 = insertvalue [2 x i64] %0, i64 %3, 1
  ret [2 x i64] %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn1I(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sI, align 8
  %3 = zext i32 %0 to i64
  %4 = shl nuw nsw i64 %3, 7
  %5 = add i64 %2, %4
  %6 = trunc i64 %5 to i32
  %7 = lshr i32 %6, 7
  %8 = and i32 %7, 1
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn2I(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i8, ptr @sI, align 8, !tbaa !10
  %3 = trunc i32 %0 to i8
  %4 = lshr i8 %2, 7
  %5 = add i8 %4, %3
  %6 = and i8 %5, 1
  %7 = zext nneg i8 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @retitI() local_unnamed_addr #3 {
  %1 = load i8, ptr @sI, align 8
  %2 = lshr i8 %1, 7
  %3 = zext nneg i8 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn3I(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i8, ptr @sI, align 8
  %3 = trunc i32 %0 to i8
  %4 = shl i8 %3, 7
  %5 = add i8 %2, %4
  store i8 %5, ptr @sI, align 8
  %6 = lshr i8 %5, 7
  %7 = zext nneg i8 %6 to i32
  ret i32 %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testI() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sI, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sI, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sI, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sI, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sI, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sI, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sI, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sI, i64 8), align 8, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = mul i32 %39, 1103515245
  %42 = add i32 %41, 12345
  store i32 %42, ptr @myrnd.s, align 4, !tbaa !6
  %43 = lshr i32 %42, 16
  %44 = trunc i32 %40 to i8
  %45 = shl i8 %44, 7
  %46 = and i8 %5, 127
  %47 = or disjoint i8 %45, %46
  store i8 %47, ptr @sI, align 8
  %48 = load i64, ptr @sI, align 8
  %49 = trunc i64 %48 to i32
  %50 = lshr i32 %49, 7
  %51 = add nuw nsw i32 %50, %43
  %52 = add nuw nsw i32 %43, %40
  %53 = xor i32 %52, %51
  %54 = and i32 %53, 1
  %55 = icmp eq i32 %54, 0
  br i1 %55, label %57, label %56

56:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

57:                                               ; preds = %0
  %58 = mul i32 %42, -2139243339
  %59 = add i32 %58, -1492899873
  %60 = lshr i32 %59, 16
  %61 = mul i32 %59, 1103515245
  %62 = add i32 %61, 12345
  store i32 %62, ptr @myrnd.s, align 4, !tbaa !6
  %63 = lshr i32 %62, 16
  %64 = trunc i32 %60 to i8
  %65 = shl i8 %64, 7
  %66 = or disjoint i8 %65, %46
  %67 = trunc i32 %63 to i8
  %68 = shl i8 %67, 7
  %69 = add i8 %66, %68
  store i8 %69, ptr @sI, align 8
  %70 = lshr i8 %69, 7
  %71 = zext nneg i8 %70 to i32
  %72 = add nuw nsw i32 %63, %60
  %73 = and i32 %72, 1
  %74 = icmp eq i32 %73, %71
  br i1 %74, label %76, label %75

75:                                               ; preds = %57
  tail call void @abort() #9
  unreachable

76:                                               ; preds = %57
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @retmeJ(i64 %0) local_unnamed_addr #1 {
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 128) i32 @fn1J(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sJ, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 9
  %5 = add i16 %4, %3
  %6 = and i16 %5, 127
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2J(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sJ, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 9
  %5 = add i16 %4, %3
  %6 = trunc i16 %5 to i8
  %7 = and i8 %6, 127
  %8 = urem i8 %7, 15
  %9 = zext nneg i8 %8 to i32
  ret i32 %9
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 128) i32 @retitJ() local_unnamed_addr #3 {
  %1 = load i16, ptr @sJ, align 4
  %2 = lshr i16 %1, 9
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 128) i32 @fn3J(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sJ, align 4
  %3 = trunc i32 %0 to i16
  %4 = shl i16 %3, 9
  %5 = add i16 %2, %4
  store i16 %5, ptr @sJ, align 4
  %6 = lshr i16 %5, 9
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testJ() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sJ, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sJ, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sJ, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sJ, i64 3), align 1, !tbaa !10
  %18 = load i16, ptr @sJ, align 4
  %19 = and i16 %18, 511
  %20 = mul i32 %15, -2139243339
  %21 = add i32 %20, -1492899873
  %22 = lshr i32 %21, 16
  %23 = mul i32 %21, 1103515245
  %24 = add i32 %23, 12345
  store i32 %24, ptr @myrnd.s, align 4, !tbaa !6
  %25 = lshr i32 %24, 16
  %26 = trunc nuw i32 %22 to i16
  %27 = shl i16 %26, 9
  %28 = or disjoint i16 %27, %19
  store i16 %28, ptr @sJ, align 4
  %29 = trunc nuw i32 %25 to i16
  %30 = add i16 %29, %26
  %31 = trunc i16 %30 to i8
  %32 = and i8 %31, 127
  %33 = urem i8 %32, 15
  %34 = add nuw nsw i32 %25, %22
  %35 = trunc i32 %34 to i8
  %36 = and i8 %35, 127
  %37 = urem i8 %36, 15
  %38 = icmp eq i8 %37, %33
  br i1 %38, label %40, label %39

39:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

40:                                               ; preds = %0
  %41 = mul i32 %24, 1103515245
  %42 = add i32 %41, 12345
  %43 = lshr i32 %42, 16
  %44 = mul i32 %42, 1103515245
  %45 = add i32 %44, 12345
  store i32 %45, ptr @myrnd.s, align 4, !tbaa !6
  %46 = lshr i32 %45, 16
  %47 = trunc nuw i32 %43 to i16
  %48 = shl i16 %47, 9
  %49 = or disjoint i16 %48, %19
  %50 = trunc nuw i32 %46 to i16
  %51 = shl i16 %50, 9
  %52 = add i16 %49, %51
  store i16 %52, ptr @sJ, align 4
  %53 = lshr i16 %52, 9
  %54 = zext nneg i16 %53 to i32
  %55 = add nuw nsw i32 %46, %43
  %56 = and i32 %55, 127
  %57 = icmp eq i32 %56, %54
  br i1 %57, label %59, label %58

58:                                               ; preds = %40
  tail call void @abort() #9
  unreachable

59:                                               ; preds = %40
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @retmeK(i64 %0) local_unnamed_addr #1 {
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn1K(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @sK, align 4, !tbaa !10
  %3 = add i32 %2, %0
  %4 = and i32 %3, 63
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2K(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @sK, align 4, !tbaa !10
  %3 = add i32 %2, %0
  %4 = trunc i32 %3 to i8
  %5 = and i8 %4, 63
  %6 = urem i8 %5, 15
  %7 = zext nneg i8 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @retitK() local_unnamed_addr #3 {
  %1 = load i32, ptr @sK, align 4
  %2 = and i32 %1, 63
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn3K(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @sK, align 4
  %3 = add i32 %2, %0
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr @sK, align 4
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @testK() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sK, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sK, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sK, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sK, i64 3), align 1, !tbaa !10
  %18 = load i32, ptr @sK, align 4
  %19 = and i32 %18, -64
  %20 = mul i32 %15, -341751747
  %21 = add i32 %20, 229283573
  %22 = lshr i32 %21, 16
  %23 = mul i32 %21, 1103515245
  %24 = add i32 %23, 12345
  store i32 %24, ptr @myrnd.s, align 4, !tbaa !6
  %25 = lshr i32 %24, 16
  %26 = add nuw nsw i32 %25, %22
  %27 = and i32 %26, 63
  %28 = or disjoint i32 %27, %19
  store i32 %28, ptr @sK, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i64 @retmeL(i64 returned %0) local_unnamed_addr #1 {
  ret i64 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn1L(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @sL, align 4, !tbaa !10
  %3 = add i32 %2, %0
  %4 = and i32 %3, 63
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2L(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @sL, align 4, !tbaa !10
  %3 = add i32 %2, %0
  %4 = trunc i32 %3 to i8
  %5 = and i8 %4, 63
  %6 = urem i8 %5, 15
  %7 = zext nneg i8 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @retitL() local_unnamed_addr #3 {
  %1 = load i32, ptr @sL, align 4
  %2 = and i32 %1, 63
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn3L(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @sL, align 4
  %3 = add i32 %2, %0
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr @sL, align 4
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @testL() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sL, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sL, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sL, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sL, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sL, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sL, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sL, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sL, i64 7), align 1, !tbaa !10
  %34 = load i32, ptr @sL, align 4
  %35 = and i32 %34, -64
  %36 = mul i32 %31, -341751747
  %37 = add i32 %36, 229283573
  %38 = lshr i32 %37, 16
  %39 = mul i32 %37, 1103515245
  %40 = add i32 %39, 12345
  store i32 %40, ptr @myrnd.s, align 4, !tbaa !6
  %41 = lshr i32 %40, 16
  %42 = add nuw nsw i32 %41, %38
  %43 = and i32 %42, 63
  %44 = or disjoint i32 %43, %35
  store i32 %44, ptr @sL, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i64 @retmeM(i64 returned %0) local_unnamed_addr #1 {
  ret i64 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn1M(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 4), align 1, !tbaa !10
  %3 = add i32 %2, %0
  %4 = and i32 %3, 63
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2M(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 4), align 1, !tbaa !10
  %3 = add i32 %2, %0
  %4 = trunc i32 %3 to i8
  %5 = and i8 %4, 63
  %6 = urem i8 %5, 15
  %7 = zext nneg i8 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @retitM() local_unnamed_addr #3 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 4), align 1
  %2 = and i32 %1, 63
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn3M(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 4), align 1
  %3 = add i32 %2, %0
  %4 = and i32 %3, 63
  %5 = and i32 %2, -64
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 4), align 1
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @testM() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sM, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 7), align 1, !tbaa !10
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 4), align 4
  %35 = and i32 %34, -64
  %36 = mul i32 %31, -341751747
  %37 = add i32 %36, 229283573
  %38 = lshr i32 %37, 16
  %39 = mul i32 %37, 1103515245
  %40 = add i32 %39, 12345
  store i32 %40, ptr @myrnd.s, align 4, !tbaa !6
  %41 = lshr i32 %40, 16
  %42 = add nuw nsw i32 %41, %38
  %43 = and i32 %42, 63
  %44 = or disjoint i32 %43, %35
  store i32 %44, ptr getelementptr inbounds nuw (i8, ptr @sM, i64 4), align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i64 @retmeN(i64 returned %0) local_unnamed_addr #1 {
  ret i64 %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn1N(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sN, align 8, !tbaa !10
  %3 = trunc i64 %2 to i32
  %4 = lshr i32 %3, 6
  %5 = add i32 %4, %0
  %6 = and i32 %5, 63
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2N(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sN, align 8, !tbaa !10
  %3 = trunc i64 %2 to i32
  %4 = lshr i32 %3, 6
  %5 = add i32 %4, %0
  %6 = trunc i32 %5 to i8
  %7 = and i8 %6, 63
  %8 = urem i8 %7, 15
  %9 = zext nneg i8 %8 to i32
  ret i32 %9
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @retitN() local_unnamed_addr #3 {
  %1 = load i64, ptr @sN, align 8
  %2 = trunc i64 %1 to i32
  %3 = lshr i32 %2, 6
  %4 = and i32 %3, 63
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 64) i32 @fn3N(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i64, ptr @sN, align 8
  %3 = trunc i64 %2 to i32
  %4 = shl i32 %0, 6
  %5 = add i32 %4, %3
  %6 = and i32 %5, 4032
  %7 = zext nneg i32 %6 to i64
  %8 = and i64 %2, -4033
  %9 = or disjoint i64 %8, %7
  store i64 %9, ptr @sN, align 8
  %10 = lshr i32 %5, 6
  %11 = and i32 %10, 63
  ret i32 %11
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testN() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sN, align 8, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sN, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sN, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sN, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sN, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sN, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sN, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sN, i64 7), align 1, !tbaa !10
  %34 = load i64, ptr @sN, align 8
  %35 = or i64 %34, 4032
  %36 = mul i32 %31, 1103515245
  %37 = add i32 %36, 12345
  %38 = lshr i32 %37, 16
  %39 = mul i32 %37, 1103515245
  %40 = add i32 %39, 12345
  store i32 %40, ptr @myrnd.s, align 4, !tbaa !6
  %41 = lshr i32 %40, 16
  %42 = shl nuw nsw i32 %38, 6
  %43 = and i32 %42, 4032
  %44 = zext nneg i32 %43 to i64
  %45 = and i64 %34, -4033
  %46 = or disjoint i64 %45, %44
  store i64 %46, ptr @sN, align 8
  %47 = trunc i64 %46 to i32
  %48 = lshr i32 %47, 6
  %49 = add nuw nsw i32 %48, %41
  %50 = xor i64 %46, %35
  %51 = icmp ult i64 %50, 4096
  br i1 %51, label %52, label %64

52:                                               ; preds = %0
  %53 = lshr exact i32 %43, 6
  %54 = and i32 %48, 63
  %55 = icmp eq i32 %53, %54
  %56 = and i64 %50, 63
  %57 = icmp eq i64 %56, 0
  %58 = and i1 %57, %55
  br i1 %58, label %59, label %64

59:                                               ; preds = %52
  %60 = add nuw nsw i32 %41, %38
  %61 = xor i32 %49, %60
  %62 = and i32 %61, 63
  %63 = icmp eq i32 %62, 0
  br i1 %63, label %65, label %64

64:                                               ; preds = %59, %52, %0
  tail call void @abort() #9
  unreachable

65:                                               ; preds = %59
  %66 = mul i32 %40, 1103515245
  %67 = add i32 %66, 12345
  %68 = lshr i32 %67, 16
  %69 = mul i32 %67, 1103515245
  %70 = add i32 %69, 12345
  store i32 %70, ptr @myrnd.s, align 4, !tbaa !6
  %71 = lshr i32 %70, 16
  %72 = shl nuw nsw i32 %68, 6
  %73 = and i32 %72, 4032
  %74 = zext nneg i32 %73 to i64
  %75 = or disjoint i64 %45, %74
  store i64 %75, ptr @sN, align 8
  %76 = trunc i64 %75 to i32
  %77 = lshr i32 %76, 6
  %78 = add nuw nsw i32 %77, %71
  %79 = trunc i32 %78 to i8
  %80 = and i8 %79, 63
  %81 = urem i8 %80, 15
  %82 = xor i64 %75, %46
  %83 = icmp ult i64 %82, 4096
  br i1 %83, label %84, label %97

84:                                               ; preds = %65
  %85 = lshr exact i32 %73, 6
  %86 = and i32 %77, 63
  %87 = icmp eq i32 %85, %86
  %88 = and i64 %82, 63
  %89 = icmp eq i64 %88, 0
  %90 = and i1 %89, %87
  br i1 %90, label %91, label %97

91:                                               ; preds = %84
  %92 = add nuw nsw i32 %71, %68
  %93 = trunc i32 %92 to i8
  %94 = and i8 %93, 63
  %95 = urem i8 %94, 15
  %96 = icmp eq i8 %95, %81
  br i1 %96, label %98, label %97

97:                                               ; preds = %91, %84, %65
  tail call void @abort() #9
  unreachable

98:                                               ; preds = %91
  %99 = mul i32 %70, 1103515245
  %100 = add i32 %99, 12345
  %101 = lshr i32 %100, 16
  %102 = mul i32 %100, 1103515245
  %103 = add i32 %102, 12345
  store i32 %103, ptr @myrnd.s, align 4, !tbaa !6
  %104 = lshr i32 %103, 16
  %105 = shl nuw nsw i32 %101, 6
  %106 = trunc i64 %45 to i32
  %107 = or i32 %105, %106
  %108 = shl nuw nsw i32 %104, 6
  %109 = and i32 %108, 131008
  %110 = add i32 %109, %107
  %111 = and i32 %110, 4032
  %112 = zext nneg i32 %111 to i64
  %113 = or disjoint i64 %45, %112
  store i64 %113, ptr @sN, align 8
  %114 = lshr i32 %110, 6
  %115 = add nuw nsw i32 %104, %101
  %116 = xor i32 %115, %114
  %117 = and i32 %116, 63
  %118 = icmp eq i32 %117, 0
  br i1 %118, label %120, label %119

119:                                              ; preds = %98
  tail call void @abort() #9
  unreachable

120:                                              ; preds = %98
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeO([2 x i64] returned %0) local_unnamed_addr #1 {
  ret [2 x i64] %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn1O(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 1, !tbaa !10
  %3 = trunc i64 %2 to i32
  %4 = add i32 %0, %3
  %5 = and i32 %4, 4095
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2O(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 1, !tbaa !10
  %3 = trunc i64 %2 to i32
  %4 = add i32 %0, %3
  %5 = trunc i32 %4 to i16
  %6 = and i16 %5, 4095
  %7 = urem i16 %6, 15
  %8 = zext nneg i16 %7 to i32
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @retitO() local_unnamed_addr #3 {
  %1 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 1
  %2 = trunc i64 %1 to i32
  %3 = and i32 %2, 4095
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn3O(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 1
  %3 = trunc i64 %2 to i32
  %4 = add i32 %0, %3
  %5 = and i32 %4, 4095
  %6 = zext nneg i32 %5 to i64
  %7 = and i64 %2, -4096
  %8 = or disjoint i64 %7, %6
  store i64 %8, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 1
  ret i32 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testO() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sO, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 4, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = trunc i32 %40 to i8
  store i8 %41, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 9), align 1, !tbaa !10
  %42 = mul i32 %39, 1103515245
  %43 = add i32 %42, 12345
  %44 = lshr i32 %43, 16
  %45 = trunc i32 %44 to i8
  store i8 %45, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 10), align 2, !tbaa !10
  %46 = mul i32 %43, 1103515245
  %47 = add i32 %46, 12345
  %48 = lshr i32 %47, 16
  %49 = trunc i32 %48 to i8
  store i8 %49, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 11), align 1, !tbaa !10
  %50 = mul i32 %47, 1103515245
  %51 = add i32 %50, 12345
  %52 = lshr i32 %51, 16
  %53 = trunc i32 %52 to i8
  store i8 %53, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 12), align 4, !tbaa !10
  %54 = mul i32 %51, 1103515245
  %55 = add i32 %54, 12345
  %56 = lshr i32 %55, 16
  %57 = trunc i32 %56 to i8
  store i8 %57, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 13), align 1, !tbaa !10
  %58 = mul i32 %55, 1103515245
  %59 = add i32 %58, 12345
  %60 = lshr i32 %59, 16
  %61 = trunc i32 %60 to i8
  store i8 %61, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 14), align 2, !tbaa !10
  %62 = mul i32 %59, 1103515245
  %63 = add i32 %62, 12345
  %64 = lshr i32 %63, 16
  %65 = trunc i32 %64 to i8
  store i8 %65, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 15), align 1, !tbaa !10
  %66 = load i64, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 4
  %67 = or i64 %66, 4095
  %68 = mul i32 %63, 1103515245
  %69 = add i32 %68, 12345
  %70 = lshr i32 %69, 16
  %71 = and i32 %70, 2047
  %72 = mul i32 %69, 1103515245
  %73 = add i32 %72, 12345
  store i32 %73, ptr @myrnd.s, align 4, !tbaa !6
  %74 = zext nneg i32 %71 to i64
  %75 = and i64 %66, -4096
  %76 = or disjoint i64 %75, %74
  store i64 %76, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 4
  %77 = xor i64 %76, %67
  %78 = icmp ult i64 %77, 34359738368
  br i1 %78, label %79, label %92

79:                                               ; preds = %0
  %80 = lshr i32 %73, 16
  %81 = and i32 %80, 2047
  %82 = trunc i64 %76 to i32
  %83 = add nuw nsw i32 %81, %82
  %84 = and i32 %83, 4095
  %85 = icmp samesign ult i64 %77, 4096
  %86 = and i32 %82, 2047
  %87 = icmp eq i32 %71, %86
  %88 = and i1 %85, %87
  %89 = add nuw nsw i32 %81, %71
  %90 = icmp eq i32 %89, %84
  %91 = select i1 %88, i1 %90, i1 false
  br i1 %91, label %93, label %92

92:                                               ; preds = %79, %0
  tail call void @abort() #9
  unreachable

93:                                               ; preds = %79
  %94 = mul i32 %73, 1103515245
  %95 = add i32 %94, 12345
  %96 = lshr i32 %95, 16
  %97 = and i32 %96, 2047
  %98 = mul i32 %95, 1103515245
  %99 = add i32 %98, 12345
  store i32 %99, ptr @myrnd.s, align 4, !tbaa !6
  %100 = lshr i32 %99, 16
  %101 = and i32 %100, 2047
  %102 = zext nneg i32 %97 to i64
  %103 = or disjoint i64 %75, %102
  store i64 %103, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 4
  %104 = trunc i64 %103 to i32
  %105 = add nuw nsw i32 %101, %104
  %106 = trunc i32 %105 to i16
  %107 = and i16 %106, 4095
  %108 = urem i16 %107, 15
  %109 = xor i64 %103, %76
  %110 = icmp ult i64 %109, 34359738368
  br i1 %110, label %111, label %121

111:                                              ; preds = %93
  %112 = icmp samesign ult i64 %109, 4096
  %113 = and i32 %104, 2047
  %114 = icmp eq i32 %97, %113
  %115 = and i1 %112, %114
  br i1 %115, label %116, label %121

116:                                              ; preds = %111
  %117 = add nuw nsw i32 %101, %97
  %118 = trunc nuw nsw i32 %117 to i16
  %119 = urem i16 %118, 15
  %120 = icmp eq i16 %119, %108
  br i1 %120, label %122, label %121

121:                                              ; preds = %116, %111, %93
  tail call void @abort() #9
  unreachable

122:                                              ; preds = %116
  %123 = mul i32 %99, 1103515245
  %124 = add i32 %123, 12345
  %125 = lshr i32 %124, 16
  %126 = and i32 %125, 2047
  %127 = mul i32 %124, 1103515245
  %128 = add i32 %127, 12345
  store i32 %128, ptr @myrnd.s, align 4, !tbaa !6
  %129 = lshr i32 %128, 16
  %130 = and i32 %129, 2047
  %131 = add nuw nsw i32 %130, %126
  %132 = zext nneg i32 %131 to i64
  %133 = or disjoint i64 %75, %132
  store i64 %133, ptr getelementptr inbounds nuw (i8, ptr @sO, i64 8), align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeP([2 x i64] returned %0) local_unnamed_addr #1 {
  ret [2 x i64] %0
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn1P(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sP, align 8, !tbaa !10
  %3 = trunc i64 %2 to i32
  %4 = add i32 %0, %3
  %5 = and i32 %4, 4095
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2P(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sP, align 8, !tbaa !10
  %3 = trunc i64 %2 to i32
  %4 = add i32 %0, %3
  %5 = trunc i32 %4 to i16
  %6 = and i16 %5, 4095
  %7 = urem i16 %6, 15
  %8 = zext nneg i16 %7 to i32
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @retitP() local_unnamed_addr #3 {
  %1 = load i64, ptr @sP, align 8
  %2 = trunc i64 %1 to i32
  %3 = and i32 %2, 4095
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn3P(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i64, ptr @sP, align 8
  %3 = trunc i64 %2 to i32
  %4 = add i32 %0, %3
  %5 = and i32 %4, 4095
  %6 = zext nneg i32 %5 to i64
  %7 = and i64 %2, -4096
  %8 = or disjoint i64 %7, %6
  store i64 %8, ptr @sP, align 8
  ret i32 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testP() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sP, align 8, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 8), align 8, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = trunc i32 %40 to i8
  store i8 %41, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 9), align 1, !tbaa !10
  %42 = mul i32 %39, 1103515245
  %43 = add i32 %42, 12345
  %44 = lshr i32 %43, 16
  %45 = trunc i32 %44 to i8
  store i8 %45, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 10), align 2, !tbaa !10
  %46 = mul i32 %43, 1103515245
  %47 = add i32 %46, 12345
  %48 = lshr i32 %47, 16
  %49 = trunc i32 %48 to i8
  store i8 %49, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 11), align 1, !tbaa !10
  %50 = mul i32 %47, 1103515245
  %51 = add i32 %50, 12345
  %52 = lshr i32 %51, 16
  %53 = trunc i32 %52 to i8
  store i8 %53, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 12), align 4, !tbaa !10
  %54 = mul i32 %51, 1103515245
  %55 = add i32 %54, 12345
  %56 = lshr i32 %55, 16
  %57 = trunc i32 %56 to i8
  store i8 %57, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 13), align 1, !tbaa !10
  %58 = mul i32 %55, 1103515245
  %59 = add i32 %58, 12345
  %60 = lshr i32 %59, 16
  %61 = trunc i32 %60 to i8
  store i8 %61, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 14), align 2, !tbaa !10
  %62 = mul i32 %59, 1103515245
  %63 = add i32 %62, 12345
  %64 = lshr i32 %63, 16
  %65 = trunc i32 %64 to i8
  store i8 %65, ptr getelementptr inbounds nuw (i8, ptr @sP, i64 15), align 1, !tbaa !10
  %66 = load i64, ptr @sP, align 8
  %67 = or i64 %66, 4095
  %68 = mul i32 %63, 1103515245
  %69 = add i32 %68, 12345
  %70 = lshr i32 %69, 16
  %71 = and i32 %70, 2047
  %72 = mul i32 %69, 1103515245
  %73 = add i32 %72, 12345
  store i32 %73, ptr @myrnd.s, align 4, !tbaa !6
  %74 = zext nneg i32 %71 to i64
  %75 = and i64 %66, -4096
  %76 = or disjoint i64 %75, %74
  store i64 %76, ptr @sP, align 8
  %77 = xor i64 %76, %67
  %78 = icmp ult i64 %77, 34359738368
  br i1 %78, label %79, label %92

79:                                               ; preds = %0
  %80 = lshr i32 %73, 16
  %81 = and i32 %80, 2047
  %82 = trunc i64 %76 to i32
  %83 = add nuw nsw i32 %81, %82
  %84 = and i32 %83, 4095
  %85 = icmp samesign ult i64 %77, 4096
  %86 = and i32 %82, 2047
  %87 = icmp eq i32 %71, %86
  %88 = and i1 %85, %87
  %89 = add nuw nsw i32 %81, %71
  %90 = icmp eq i32 %89, %84
  %91 = select i1 %88, i1 %90, i1 false
  br i1 %91, label %93, label %92

92:                                               ; preds = %79, %0
  tail call void @abort() #9
  unreachable

93:                                               ; preds = %79
  %94 = mul i32 %73, 1103515245
  %95 = add i32 %94, 12345
  %96 = lshr i32 %95, 16
  %97 = and i32 %96, 2047
  %98 = mul i32 %95, 1103515245
  %99 = add i32 %98, 12345
  store i32 %99, ptr @myrnd.s, align 4, !tbaa !6
  %100 = lshr i32 %99, 16
  %101 = and i32 %100, 2047
  %102 = zext nneg i32 %97 to i64
  %103 = or disjoint i64 %75, %102
  store i64 %103, ptr @sP, align 8
  %104 = trunc i64 %103 to i32
  %105 = add nuw nsw i32 %101, %104
  %106 = trunc i32 %105 to i16
  %107 = and i16 %106, 4095
  %108 = urem i16 %107, 15
  %109 = xor i64 %103, %76
  %110 = icmp ult i64 %109, 34359738368
  br i1 %110, label %111, label %121

111:                                              ; preds = %93
  %112 = icmp samesign ult i64 %109, 4096
  %113 = and i32 %104, 2047
  %114 = icmp eq i32 %97, %113
  %115 = and i1 %112, %114
  br i1 %115, label %116, label %121

116:                                              ; preds = %111
  %117 = add nuw nsw i32 %101, %97
  %118 = trunc nuw nsw i32 %117 to i16
  %119 = urem i16 %118, 15
  %120 = icmp eq i16 %119, %108
  br i1 %120, label %122, label %121

121:                                              ; preds = %116, %111, %93
  tail call void @abort() #9
  unreachable

122:                                              ; preds = %116
  %123 = mul i32 %99, 1103515245
  %124 = add i32 %123, 12345
  %125 = lshr i32 %124, 16
  %126 = and i32 %125, 2047
  %127 = mul i32 %124, 1103515245
  %128 = add i32 %127, 12345
  store i32 %128, ptr @myrnd.s, align 4, !tbaa !6
  %129 = lshr i32 %128, 16
  %130 = and i32 %129, 2047
  %131 = add nuw nsw i32 %130, %126
  %132 = zext nneg i32 %131 to i64
  %133 = or disjoint i64 %75, %132
  store i64 %133, ptr @sP, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeQ([2 x i64] %0) local_unnamed_addr #1 {
  %2 = extractvalue [2 x i64] %0, 1
  %3 = and i64 %2, 65535
  %4 = insertvalue [2 x i64] %0, i64 %3, 1
  ret [2 x i64] %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn1Q(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sQ, align 8
  %3 = trunc i64 %2 to i32
  %4 = add i32 %0, %3
  %5 = and i32 %4, 4095
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2Q(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sQ, align 8, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = add i16 %2, %3
  %5 = and i16 %4, 4095
  %6 = urem i16 %5, 15
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @retitQ() local_unnamed_addr #3 {
  %1 = load i16, ptr @sQ, align 8
  %2 = and i16 %1, 4095
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn3Q(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sQ, align 8
  %3 = trunc i32 %0 to i16
  %4 = add i16 %2, %3
  %5 = and i16 %4, 4095
  %6 = and i16 %2, -4096
  %7 = or disjoint i16 %5, %6
  store i16 %7, ptr @sQ, align 8
  %8 = zext nneg i16 %5 to i32
  ret i32 %8
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testQ() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sQ, align 8, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sQ, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sQ, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sQ, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sQ, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sQ, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sQ, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sQ, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sQ, i64 8), align 8, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = trunc i32 %40 to i8
  store i8 %41, ptr getelementptr inbounds nuw (i8, ptr @sQ, i64 9), align 1, !tbaa !10
  %42 = load i16, ptr @sQ, align 8
  %43 = mul i32 %39, 1103515245
  %44 = add i32 %43, 12345
  %45 = lshr i32 %44, 16
  %46 = and i32 %45, 2047
  %47 = mul i32 %44, 1103515245
  %48 = add i32 %47, 12345
  store i32 %48, ptr @myrnd.s, align 4, !tbaa !6
  %49 = trunc nuw nsw i32 %46 to i16
  %50 = and i16 %42, -4096
  %51 = or disjoint i16 %50, %49
  store i16 %51, ptr @sQ, align 8
  %52 = lshr i32 %48, 16
  %53 = and i32 %52, 2047
  %54 = load i64, ptr @sQ, align 8
  %55 = trunc i64 %54 to i32
  %56 = add i32 %53, %55
  %57 = and i32 %56, 4095
  %58 = add nuw nsw i32 %53, %46
  %59 = icmp eq i32 %58, %57
  br i1 %59, label %61, label %60

60:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

61:                                               ; preds = %0
  %62 = mul i32 %48, 1103515245
  %63 = add i32 %62, 12345
  %64 = lshr i32 %63, 16
  %65 = and i32 %64, 2047
  %66 = mul i32 %63, 1103515245
  %67 = add i32 %66, 12345
  store i32 %67, ptr @myrnd.s, align 4, !tbaa !6
  %68 = lshr i32 %67, 16
  %69 = and i32 %68, 2047
  %70 = trunc nuw nsw i32 %65 to i16
  %71 = or disjoint i16 %50, %70
  store i16 %71, ptr @sQ, align 8
  %72 = trunc nuw nsw i32 %69 to i16
  %73 = add nuw nsw i16 %72, %70
  %74 = urem i16 %73, 15
  %75 = add nuw nsw i32 %69, %65
  %76 = trunc nuw nsw i32 %75 to i16
  %77 = urem i16 %76, 15
  %78 = icmp eq i16 %77, %74
  br i1 %78, label %80, label %79

79:                                               ; preds = %61
  tail call void @abort() #9
  unreachable

80:                                               ; preds = %61
  %81 = mul i32 %67, 1103515245
  %82 = add i32 %81, 12345
  %83 = lshr i32 %82, 16
  %84 = mul i32 %82, 1103515245
  %85 = add i32 %84, 12345
  store i32 %85, ptr @myrnd.s, align 4, !tbaa !6
  %86 = lshr i32 %85, 16
  %87 = trunc nuw i32 %83 to i16
  %88 = and i16 %87, 2047
  %89 = trunc nuw i32 %86 to i16
  %90 = and i16 %89, 2047
  %91 = add nuw nsw i16 %90, %88
  %92 = or disjoint i16 %91, %50
  store i16 %92, ptr @sQ, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeR([2 x i64] %0) local_unnamed_addr #1 {
  %2 = extractvalue [2 x i64] %0, 1
  %3 = and i64 %2, 65535
  %4 = insertvalue [2 x i64] %0, i64 %3, 1
  ret [2 x i64] %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4) i32 @fn1R(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sR, align 8
  %3 = trunc i64 %2 to i32
  %4 = add i32 %0, %3
  %5 = and i32 %4, 3
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4) i32 @fn2R(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sR, align 8, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = add i16 %2, %3
  %5 = and i16 %4, 3
  %6 = zext nneg i16 %5 to i32
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4) i32 @retitR() local_unnamed_addr #3 {
  %1 = load i16, ptr @sR, align 8
  %2 = and i16 %1, 3
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4) i32 @fn3R(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sR, align 8
  %3 = trunc i32 %0 to i16
  %4 = add i16 %2, %3
  %5 = and i16 %4, 3
  %6 = and i16 %2, -4
  %7 = or disjoint i16 %5, %6
  store i16 %7, ptr @sR, align 8
  %8 = zext nneg i16 %5 to i32
  ret i32 %8
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testR() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sR, align 8, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sR, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sR, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sR, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sR, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sR, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sR, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sR, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sR, i64 8), align 8, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = trunc i32 %40 to i8
  store i8 %41, ptr getelementptr inbounds nuw (i8, ptr @sR, i64 9), align 1, !tbaa !10
  %42 = load i16, ptr @sR, align 8
  %43 = mul i32 %39, 1103515245
  %44 = add i32 %43, 12345
  %45 = lshr i32 %44, 16
  %46 = mul i32 %44, 1103515245
  %47 = add i32 %46, 12345
  store i32 %47, ptr @myrnd.s, align 4, !tbaa !6
  %48 = lshr i32 %47, 16
  %49 = trunc nuw i32 %45 to i16
  %50 = and i16 %49, 3
  %51 = and i16 %42, -4
  %52 = or disjoint i16 %50, %51
  store i16 %52, ptr @sR, align 8
  %53 = load i64, ptr @sR, align 8
  %54 = trunc i64 %53 to i32
  %55 = add i32 %48, %54
  %56 = add nuw nsw i32 %48, %45
  %57 = xor i32 %56, %55
  %58 = and i32 %57, 3
  %59 = icmp eq i32 %58, 0
  br i1 %59, label %61, label %60

60:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

61:                                               ; preds = %0
  %62 = mul i32 %47, -2139243339
  %63 = add i32 %62, -1492899873
  %64 = lshr i32 %63, 16
  %65 = mul i32 %63, 1103515245
  %66 = add i32 %65, 12345
  store i32 %66, ptr @myrnd.s, align 4, !tbaa !6
  %67 = lshr i32 %66, 16
  %68 = trunc nuw i32 %64 to i16
  %69 = trunc nuw i32 %67 to i16
  %70 = add i16 %69, %68
  %71 = and i16 %70, 3
  %72 = or disjoint i16 %71, %51
  store i16 %72, ptr @sR, align 8
  %73 = zext nneg i16 %71 to i32
  %74 = add nuw nsw i32 %67, %64
  %75 = and i32 %74, 3
  %76 = icmp eq i32 %75, %73
  br i1 %76, label %78, label %77

77:                                               ; preds = %61
  tail call void @abort() #9
  unreachable

78:                                               ; preds = %61
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeS([2 x i64] %0) local_unnamed_addr #1 {
  %2 = extractvalue [2 x i64] %0, 1
  %3 = and i64 %2, 65535
  %4 = insertvalue [2 x i64] %0, i64 %3, 1
  ret [2 x i64] %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn1S(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sS, align 8
  %3 = trunc i64 %2 to i32
  %4 = add i32 %0, %3
  %5 = and i32 %4, 1
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn2S(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sS, align 8, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = add i16 %2, %3
  %5 = and i16 %4, 1
  %6 = zext nneg i16 %5 to i32
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @retitS() local_unnamed_addr #3 {
  %1 = load i16, ptr @sS, align 8
  %2 = and i16 %1, 1
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn3S(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sS, align 8
  %3 = trunc i32 %0 to i16
  %4 = add i16 %2, %3
  %5 = and i16 %4, 1
  %6 = and i16 %2, -2
  %7 = or disjoint i16 %5, %6
  store i16 %7, ptr @sS, align 8
  %8 = zext nneg i16 %5 to i32
  ret i32 %8
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testS() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sS, align 8, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sS, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sS, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sS, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sS, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sS, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sS, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sS, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sS, i64 8), align 8, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = trunc i32 %40 to i8
  store i8 %41, ptr getelementptr inbounds nuw (i8, ptr @sS, i64 9), align 1, !tbaa !10
  %42 = load i16, ptr @sS, align 8
  %43 = mul i32 %39, 1103515245
  %44 = add i32 %43, 12345
  %45 = lshr i32 %44, 16
  %46 = mul i32 %44, 1103515245
  %47 = add i32 %46, 12345
  store i32 %47, ptr @myrnd.s, align 4, !tbaa !6
  %48 = lshr i32 %47, 16
  %49 = trunc nuw i32 %45 to i16
  %50 = and i16 %49, 1
  %51 = and i16 %42, -2
  %52 = or disjoint i16 %50, %51
  store i16 %52, ptr @sS, align 8
  %53 = load i64, ptr @sS, align 8
  %54 = trunc i64 %53 to i32
  %55 = add i32 %48, %54
  %56 = add nuw nsw i32 %48, %45
  %57 = xor i32 %56, %55
  %58 = and i32 %57, 1
  %59 = icmp eq i32 %58, 0
  br i1 %59, label %61, label %60

60:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

61:                                               ; preds = %0
  %62 = mul i32 %47, -2139243339
  %63 = add i32 %62, -1492899873
  %64 = lshr i32 %63, 16
  %65 = mul i32 %63, 1103515245
  %66 = add i32 %65, 12345
  store i32 %66, ptr @myrnd.s, align 4, !tbaa !6
  %67 = lshr i32 %66, 16
  %68 = trunc nuw i32 %64 to i16
  %69 = trunc nuw i32 %67 to i16
  %70 = add i16 %69, %68
  %71 = and i16 %70, 1
  %72 = or disjoint i16 %71, %51
  store i16 %72, ptr @sS, align 8
  %73 = zext nneg i16 %71 to i32
  %74 = add nuw nsw i32 %67, %64
  %75 = and i32 %74, 1
  %76 = icmp eq i32 %75, %73
  br i1 %76, label %78, label %77

77:                                               ; preds = %61
  tail call void @abort() #9
  unreachable

78:                                               ; preds = %61
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @retmeT(i64 %0) local_unnamed_addr #1 {
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn1T(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sT, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = add i16 %2, %3
  %5 = and i16 %4, 1
  %6 = zext nneg i16 %5 to i32
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn2T(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sT, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = add i16 %2, %3
  %5 = and i16 %4, 1
  %6 = zext nneg i16 %5 to i32
  ret i32 %6
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @retitT() local_unnamed_addr #3 {
  %1 = load i16, ptr @sT, align 4
  %2 = and i16 %1, 1
  %3 = zext nneg i16 %2 to i32
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn3T(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sT, align 4
  %3 = trunc i32 %0 to i16
  %4 = add i16 %2, %3
  %5 = and i16 %4, 1
  %6 = and i16 %2, -2
  %7 = or disjoint i16 %5, %6
  store i16 %7, ptr @sT, align 4
  %8 = zext nneg i16 %5 to i32
  ret i32 %8
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testT() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sT, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sT, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sT, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sT, i64 3), align 1, !tbaa !10
  %18 = load i16, ptr @sT, align 4
  %19 = and i16 %18, -2
  %20 = mul i32 %15, -341751747
  %21 = add i32 %20, 229283573
  %22 = lshr i32 %21, 16
  %23 = mul i32 %21, 1103515245
  %24 = add i32 %23, 12345
  store i32 %24, ptr @myrnd.s, align 4, !tbaa !6
  %25 = lshr i32 %24, 16
  %26 = trunc nuw i32 %22 to i16
  %27 = trunc nuw i32 %25 to i16
  %28 = add i16 %27, %26
  %29 = and i16 %28, 1
  %30 = or disjoint i16 %29, %19
  store i16 %30, ptr @sT, align 4
  %31 = zext nneg i16 %29 to i32
  %32 = add nuw nsw i32 %25, %22
  %33 = and i32 %32, 1
  %34 = icmp eq i32 %33, %31
  br i1 %34, label %36, label %35

35:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

36:                                               ; preds = %0
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local [2 x i64] @retmeU([2 x i64] %0) local_unnamed_addr #1 {
  %2 = extractvalue [2 x i64] %0, 1
  %3 = and i64 %2, 65535
  %4 = insertvalue [2 x i64] %0, i64 %3, 1
  ret [2 x i64] %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn1U(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i64, ptr @sU, align 8
  %3 = zext i32 %0 to i64
  %4 = shl nuw nsw i64 %3, 6
  %5 = add i64 %2, %4
  %6 = trunc i64 %5 to i32
  %7 = lshr i32 %6, 6
  %8 = and i32 %7, 1
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn2U(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sU, align 8, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 6
  %5 = add i16 %4, %3
  %6 = and i16 %5, 1
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @retitU() local_unnamed_addr #3 {
  %1 = load i16, ptr @sU, align 8
  %2 = lshr i16 %1, 6
  %3 = and i16 %2, 1
  %4 = zext nneg i16 %3 to i32
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn3U(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sU, align 8
  %3 = trunc i32 %0 to i16
  %4 = shl i16 %3, 6
  %5 = add i16 %2, %4
  %6 = and i16 %5, 64
  %7 = and i16 %2, -65
  %8 = or disjoint i16 %6, %7
  store i16 %8, ptr @sU, align 8
  %9 = lshr i16 %5, 6
  %10 = and i16 %9, 1
  %11 = zext nneg i16 %10 to i32
  ret i32 %11
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testU() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sU, align 8, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sU, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sU, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sU, i64 3), align 1, !tbaa !10
  %18 = mul i32 %15, 1103515245
  %19 = add i32 %18, 12345
  %20 = lshr i32 %19, 16
  %21 = trunc i32 %20 to i8
  store i8 %21, ptr getelementptr inbounds nuw (i8, ptr @sU, i64 4), align 4, !tbaa !10
  %22 = mul i32 %19, 1103515245
  %23 = add i32 %22, 12345
  %24 = lshr i32 %23, 16
  %25 = trunc i32 %24 to i8
  store i8 %25, ptr getelementptr inbounds nuw (i8, ptr @sU, i64 5), align 1, !tbaa !10
  %26 = mul i32 %23, 1103515245
  %27 = add i32 %26, 12345
  %28 = lshr i32 %27, 16
  %29 = trunc i32 %28 to i8
  store i8 %29, ptr getelementptr inbounds nuw (i8, ptr @sU, i64 6), align 2, !tbaa !10
  %30 = mul i32 %27, 1103515245
  %31 = add i32 %30, 12345
  %32 = lshr i32 %31, 16
  %33 = trunc i32 %32 to i8
  store i8 %33, ptr getelementptr inbounds nuw (i8, ptr @sU, i64 7), align 1, !tbaa !10
  %34 = mul i32 %31, 1103515245
  %35 = add i32 %34, 12345
  %36 = lshr i32 %35, 16
  %37 = trunc i32 %36 to i8
  store i8 %37, ptr getelementptr inbounds nuw (i8, ptr @sU, i64 8), align 8, !tbaa !10
  %38 = mul i32 %35, 1103515245
  %39 = add i32 %38, 12345
  %40 = lshr i32 %39, 16
  %41 = trunc i32 %40 to i8
  store i8 %41, ptr getelementptr inbounds nuw (i8, ptr @sU, i64 9), align 1, !tbaa !10
  %42 = load i16, ptr @sU, align 8
  %43 = mul i32 %39, 1103515245
  %44 = add i32 %43, 12345
  %45 = lshr i32 %44, 16
  %46 = mul i32 %44, 1103515245
  %47 = add i32 %46, 12345
  store i32 %47, ptr @myrnd.s, align 4, !tbaa !6
  %48 = lshr i32 %47, 16
  %49 = trunc nuw i32 %45 to i16
  %50 = shl i16 %49, 6
  %51 = and i16 %50, 64
  %52 = and i16 %42, -65
  %53 = or disjoint i16 %51, %52
  store i16 %53, ptr @sU, align 8
  %54 = load i64, ptr @sU, align 8
  %55 = trunc i64 %54 to i32
  %56 = lshr i32 %55, 6
  %57 = add nuw nsw i32 %56, %48
  %58 = add nuw nsw i32 %48, %45
  %59 = xor i32 %58, %57
  %60 = and i32 %59, 1
  %61 = icmp eq i32 %60, 0
  br i1 %61, label %63, label %62

62:                                               ; preds = %0
  tail call void @abort() #9
  unreachable

63:                                               ; preds = %0
  %64 = mul i32 %47, 1103515245
  %65 = add i32 %64, 12345
  %66 = lshr i32 %65, 16
  %67 = mul i32 %65, 1103515245
  %68 = add i32 %67, 12345
  store i32 %68, ptr @myrnd.s, align 4, !tbaa !6
  %69 = trunc nuw i32 %66 to i16
  %70 = shl i16 %69, 6
  %71 = and i16 %70, 64
  %72 = or disjoint i16 %71, %52
  store i16 %72, ptr @sU, align 8
  %73 = lshr i16 %72, 6
  %74 = and i32 %66, 1
  %75 = and i16 %73, 1
  %76 = zext nneg i16 %75 to i32
  %77 = icmp eq i32 %74, %76
  br i1 %77, label %78, label %87

78:                                               ; preds = %63
  %79 = lshr i32 %68, 16
  %80 = trunc nuw i32 %79 to i16
  %81 = add i16 %73, %80
  %82 = and i16 %81, 1
  %83 = zext nneg i16 %82 to i32
  %84 = add nuw nsw i32 %79, %66
  %85 = and i32 %84, 1
  %86 = icmp eq i32 %85, %83
  br i1 %86, label %88, label %87

87:                                               ; preds = %78, %63
  tail call void @abort() #9
  unreachable

88:                                               ; preds = %78
  %89 = mul i32 %68, 1103515245
  %90 = add i32 %89, 12345
  %91 = lshr i32 %90, 16
  %92 = mul i32 %90, 1103515245
  %93 = add i32 %92, 12345
  store i32 %93, ptr @myrnd.s, align 4, !tbaa !6
  %94 = lshr i32 %93, 16
  %95 = trunc nuw i32 %91 to i16
  %96 = shl i16 %95, 6
  %97 = or i16 %96, %52
  %98 = trunc nuw i32 %94 to i16
  %99 = shl i16 %98, 6
  %100 = add i16 %97, %99
  %101 = and i16 %100, 64
  %102 = or disjoint i16 %101, %52
  store i16 %102, ptr @sU, align 8
  %103 = lshr i16 %100, 6
  %104 = and i16 %103, 1
  %105 = zext nneg i16 %104 to i32
  %106 = add nuw nsw i32 %94, %91
  %107 = and i32 %106, 1
  %108 = icmp eq i32 %107, %105
  br i1 %108, label %110, label %109

109:                                              ; preds = %88
  tail call void @abort() #9
  unreachable

110:                                              ; preds = %88
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local i32 @retmeV(i64 %0) local_unnamed_addr #1 {
  %2 = trunc i64 %0 to i32
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn1V(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sV, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 8
  %5 = add i16 %4, %3
  %6 = and i16 %5, 1
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn2V(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i16, ptr @sV, align 4, !tbaa !10
  %3 = trunc i32 %0 to i16
  %4 = lshr i16 %2, 8
  %5 = add i16 %4, %3
  %6 = and i16 %5, 1
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @retitV() local_unnamed_addr #3 {
  %1 = load i16, ptr @sV, align 4
  %2 = lshr i16 %1, 8
  %3 = and i16 %2, 1
  %4 = zext nneg i16 %3 to i32
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @fn3V(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i16, ptr @sV, align 4
  %3 = trunc i32 %0 to i16
  %4 = shl i16 %3, 8
  %5 = add i16 %2, %4
  %6 = and i16 %5, 256
  %7 = and i16 %2, -257
  %8 = or disjoint i16 %6, %7
  store i16 %8, ptr @sV, align 4
  %9 = lshr i16 %5, 8
  %10 = and i16 %9, 1
  %11 = zext nneg i16 %10 to i32
  ret i32 %11
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testV() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sV, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sV, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sV, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sV, i64 3), align 1, !tbaa !10
  %18 = load i16, ptr @sV, align 4
  %19 = mul i32 %15, 1103515245
  %20 = add i32 %19, 12345
  %21 = lshr i32 %20, 16
  %22 = mul i32 %20, 1103515245
  %23 = add i32 %22, 12345
  store i32 %23, ptr @myrnd.s, align 4, !tbaa !6
  %24 = trunc nuw i32 %21 to i16
  %25 = shl i16 %24, 8
  %26 = and i16 %25, 256
  %27 = and i16 %18, -257
  %28 = or disjoint i16 %26, %27
  store i16 %28, ptr @sV, align 4
  %29 = lshr i16 %28, 8
  %30 = and i32 %21, 1
  %31 = and i16 %29, 1
  %32 = zext nneg i16 %31 to i32
  %33 = icmp eq i32 %30, %32
  br i1 %33, label %34, label %43

34:                                               ; preds = %0
  %35 = lshr i32 %23, 16
  %36 = trunc nuw i32 %35 to i16
  %37 = add i16 %29, %36
  %38 = and i16 %37, 1
  %39 = zext nneg i16 %38 to i32
  %40 = add nuw nsw i32 %35, %21
  %41 = and i32 %40, 1
  %42 = icmp eq i32 %41, %39
  br i1 %42, label %44, label %43

43:                                               ; preds = %34, %0
  tail call void @abort() #9
  unreachable

44:                                               ; preds = %34
  %45 = mul i32 %23, 1103515245
  %46 = add i32 %45, 12345
  %47 = lshr i32 %46, 16
  %48 = mul i32 %46, 1103515245
  %49 = add i32 %48, 12345
  store i32 %49, ptr @myrnd.s, align 4, !tbaa !6
  %50 = trunc nuw i32 %47 to i16
  %51 = shl i16 %50, 8
  %52 = and i16 %51, 256
  %53 = or disjoint i16 %52, %27
  store i16 %53, ptr @sV, align 4
  %54 = lshr i16 %53, 8
  %55 = and i32 %47, 1
  %56 = and i16 %54, 1
  %57 = zext nneg i16 %56 to i32
  %58 = icmp eq i32 %55, %57
  br i1 %58, label %59, label %68

59:                                               ; preds = %44
  %60 = lshr i32 %49, 16
  %61 = trunc nuw i32 %60 to i16
  %62 = add i16 %54, %61
  %63 = and i16 %62, 1
  %64 = zext nneg i16 %63 to i32
  %65 = add nuw nsw i32 %60, %47
  %66 = and i32 %65, 1
  %67 = icmp eq i32 %66, %64
  br i1 %67, label %69, label %68

68:                                               ; preds = %59, %44
  tail call void @abort() #9
  unreachable

69:                                               ; preds = %59
  %70 = mul i32 %49, 1103515245
  %71 = add i32 %70, 12345
  %72 = lshr i32 %71, 16
  %73 = mul i32 %71, 1103515245
  %74 = add i32 %73, 12345
  store i32 %74, ptr @myrnd.s, align 4, !tbaa !6
  %75 = lshr i32 %74, 16
  %76 = trunc nuw i32 %72 to i16
  %77 = shl i16 %76, 8
  %78 = or i16 %77, %27
  %79 = trunc nuw i32 %75 to i16
  %80 = shl i16 %79, 8
  %81 = add i16 %78, %80
  %82 = and i16 %81, 256
  %83 = or disjoint i16 %82, %27
  store i16 %83, ptr @sV, align 4
  %84 = lshr i16 %81, 8
  %85 = and i16 %84, 1
  %86 = zext nneg i16 %85 to i32
  %87 = add nuw nsw i32 %75, %72
  %88 = and i32 %87, 1
  %89 = icmp eq i32 %88, %86
  br i1 %89, label %91, label %90

90:                                               ; preds = %69
  tail call void @abort() #9
  unreachable

91:                                               ; preds = %69
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @retmeW(ptr dead_on_unwind noalias writable writeonly sret(%struct.W) align 1 captures(none) initializes((0, 20)) %0, ptr dead_on_return noundef readonly captures(none) %1) local_unnamed_addr #6 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(20) %0, ptr noundef nonnull align 1 dereferenceable(20) %1, i64 20, i1 false), !tbaa.struct !11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn1W(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 16), align 1, !tbaa !10
  %3 = add i32 %2, %0
  %4 = and i32 %3, 4095
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2W(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 16), align 1, !tbaa !10
  %3 = add i32 %2, %0
  %4 = trunc i32 %3 to i16
  %5 = and i16 %4, 4095
  %6 = urem i16 %5, 15
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @retitW() local_unnamed_addr #3 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 16), align 1
  %2 = and i32 %1, 4095
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn3W(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 16), align 1
  %3 = add i32 %2, %0
  %4 = and i32 %3, 4095
  %5 = and i32 %2, -4096
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 16), align 1
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @testW() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1670464429
  %3 = add i32 %2, 2121308585
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 16), align 16, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 17), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 18), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 19), align 1, !tbaa !10
  store fp128 0xL00000000000000004001500000000000, ptr @sW, align 16, !tbaa !14
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 16), align 16
  %19 = and i32 %18, -4096
  %20 = mul i32 %15, -341751747
  %21 = add i32 %20, 229283573
  %22 = lshr i32 %21, 16
  %23 = and i32 %22, 2047
  %24 = mul i32 %21, 1103515245
  %25 = add i32 %24, 12345
  store i32 %25, ptr @myrnd.s, align 4, !tbaa !6
  %26 = lshr i32 %25, 16
  %27 = and i32 %26, 2047
  %28 = add nuw nsw i32 %27, %23
  %29 = or disjoint i32 %28, %19
  store i32 %29, ptr getelementptr inbounds nuw (i8, ptr @sW, i64 16), align 16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @retmeX(ptr dead_on_unwind noalias writable writeonly sret(%struct.X) align 1 captures(none) initializes((0, 20)) %0, ptr dead_on_return noundef readonly captures(none) %1) local_unnamed_addr #6 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(20) %0, ptr noundef nonnull align 1 dereferenceable(20) %1, i64 20, i1 false), !tbaa.struct !16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn1X(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @sX, align 4, !tbaa !10
  %3 = add i32 %2, %0
  %4 = and i32 %3, 4095
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2X(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @sX, align 4, !tbaa !10
  %3 = add i32 %2, %0
  %4 = trunc i32 %3 to i16
  %5 = and i16 %4, 4095
  %6 = urem i16 %5, 15
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @retitX() local_unnamed_addr #3 {
  %1 = load i32, ptr @sX, align 4
  %2 = and i32 %1, 4095
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn3X(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @sX, align 4
  %3 = add i32 %2, %0
  %4 = and i32 %3, 4095
  %5 = and i32 %2, -4096
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr @sX, align 4
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @testX() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sX, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sX, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sX, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sX, i64 3), align 1, !tbaa !10
  store fp128 0xL00000000000000004001500000000000, ptr getelementptr inbounds nuw (i8, ptr @sX, i64 4), align 4, !tbaa !17
  %18 = load i32, ptr @sX, align 4
  %19 = and i32 %18, -4096
  %20 = mul i32 %15, 424038781
  %21 = add i32 %20, -804247707
  %22 = lshr i32 %21, 16
  %23 = and i32 %22, 2047
  %24 = mul i32 %21, 1103515245
  %25 = add i32 %24, 12345
  store i32 %25, ptr @myrnd.s, align 4, !tbaa !6
  %26 = lshr i32 %25, 16
  %27 = and i32 %26, 2047
  %28 = add nuw nsw i32 %27, %23
  %29 = or disjoint i32 %28, %19
  store i32 %29, ptr @sX, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @retmeY(ptr dead_on_unwind noalias writable writeonly sret(%struct.Y) align 1 captures(none) initializes((0, 20)) %0, ptr dead_on_return noundef readonly captures(none) %1) local_unnamed_addr #6 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(20) %0, ptr noundef nonnull align 1 dereferenceable(20) %1, i64 20, i1 false), !tbaa.struct !16
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn1Y(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @sY, align 4, !tbaa !10
  %3 = add i32 %2, %0
  %4 = and i32 %3, 4095
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2Y(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr @sY, align 4, !tbaa !10
  %3 = add i32 %2, %0
  %4 = trunc i32 %3 to i16
  %5 = and i16 %4, 4095
  %6 = urem i16 %5, 15
  %7 = zext nneg i16 %6 to i32
  ret i32 %7
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @retitY() local_unnamed_addr #3 {
  %1 = load i32, ptr @sY, align 4
  %2 = and i32 %1, 4095
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn3Y(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr @sY, align 4
  %3 = add i32 %2, %0
  %4 = and i32 %3, 4095
  %5 = and i32 %2, -4096
  %6 = or disjoint i32 %4, %5
  store i32 %6, ptr @sY, align 4
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @testY() local_unnamed_addr #0 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1103515245
  %3 = add i32 %2, 12345
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr @sY, align 4, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sY, i64 1), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sY, i64 2), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sY, i64 3), align 1, !tbaa !10
  store fp128 0xL00000000000000004001500000000000, ptr getelementptr inbounds nuw (i8, ptr @sY, i64 4), align 4, !tbaa !19
  %18 = load i32, ptr @sY, align 4
  %19 = and i32 %18, -4096
  %20 = mul i32 %15, 424038781
  %21 = add i32 %20, -804247707
  %22 = lshr i32 %21, 16
  %23 = and i32 %22, 2047
  %24 = mul i32 %21, 1103515245
  %25 = add i32 %24, 12345
  store i32 %25, ptr @myrnd.s, align 4, !tbaa !6
  %26 = lshr i32 %25, 16
  %27 = and i32 %26, 2047
  %28 = add nuw nsw i32 %27, %23
  %29 = or disjoint i32 %28, %19
  store i32 %29, ptr @sY, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable
define dso_local void @retmeZ(ptr dead_on_unwind noalias writable writeonly sret(%struct.Z) align 1 captures(none) initializes((0, 20)) %0, ptr dead_on_return noundef readonly captures(none) %1) local_unnamed_addr #6 {
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(20) %0, ptr noundef nonnull align 1 dereferenceable(20) %1, i64 20, i1 false), !tbaa.struct !11
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn1Z(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 16), align 1, !tbaa !10
  %3 = lshr i32 %2, 20
  %4 = add i32 %3, %0
  %5 = and i32 %4, 4095
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 15) i32 @fn2Z(i32 noundef %0) local_unnamed_addr #3 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 16), align 1, !tbaa !10
  %3 = lshr i32 %2, 20
  %4 = add i32 %3, %0
  %5 = trunc i32 %4 to i16
  %6 = and i16 %5, 4095
  %7 = urem i16 %6, 15
  %8 = zext nneg i16 %7 to i32
  ret i32 %8
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @retitZ() local_unnamed_addr #3 {
  %1 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 16), align 1
  %2 = lshr i32 %1, 20
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 4096) i32 @fn3Z(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 16), align 1
  %3 = shl i32 %0, 20
  %4 = add i32 %2, %3
  store i32 %4, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 16), align 1
  %5 = lshr i32 %4, 20
  ret i32 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @testZ() local_unnamed_addr #4 {
  %1 = load i32, ptr @myrnd.s, align 4, !tbaa !6
  %2 = mul i32 %1, 1670464429
  %3 = add i32 %2, 2121308585
  %4 = lshr i32 %3, 16
  %5 = trunc i32 %4 to i8
  store i8 %5, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 16), align 16, !tbaa !10
  %6 = mul i32 %3, 1103515245
  %7 = add i32 %6, 12345
  %8 = lshr i32 %7, 16
  %9 = trunc i32 %8 to i8
  store i8 %9, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 17), align 1, !tbaa !10
  %10 = mul i32 %7, 1103515245
  %11 = add i32 %10, 12345
  %12 = lshr i32 %11, 16
  %13 = trunc i32 %12 to i8
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 18), align 2, !tbaa !10
  %14 = mul i32 %11, 1103515245
  %15 = add i32 %14, 12345
  %16 = lshr i32 %15, 16
  %17 = trunc i32 %16 to i8
  store i8 %17, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 19), align 1, !tbaa !10
  store fp128 0xL00000000000000004001500000000000, ptr @sZ, align 16, !tbaa !21
  %18 = load i32, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 16), align 16
  %19 = and i32 %18, 1048575
  %20 = mul i32 %15, -2139243339
  %21 = add i32 %20, -1492899873
  %22 = shl i32 %21, 4
  %23 = and i32 %22, 2146435072
  %24 = or disjoint i32 %23, %19
  %25 = mul i32 %21, -1029531031
  %26 = add i32 %25, -740551042
  %27 = lshr i32 %26, 16
  %28 = and i32 %27, 2047
  %29 = mul i32 %26, 1103515245
  %30 = add i32 %29, 12345
  store i32 %30, ptr @myrnd.s, align 4, !tbaa !6
  %31 = lshr i32 %30, 16
  %32 = and i32 %31, 2047
  %33 = shl nuw nsw i32 %28, 20
  %34 = or disjoint i32 %33, %19
  %35 = shl nuw nsw i32 %32, 20
  %36 = add nuw i32 %34, %35
  store i32 %36, ptr getelementptr inbounds nuw (i8, ptr @sZ, i64 16), align 16
  %37 = xor i32 %36, %24
  %38 = and i32 %37, 1040384
  %39 = icmp eq i32 %38, 0
  br i1 %39, label %40, label %47

40:                                               ; preds = %0
  %41 = lshr i32 %36, 20
  %42 = and i32 %37, 8191
  %43 = icmp eq i32 %42, 0
  %44 = add nuw nsw i32 %32, %28
  %45 = icmp eq i32 %44, %41
  %46 = select i1 %43, i1 %45, i1 false
  br i1 %46, label %48, label %47

47:                                               ; preds = %40, %0
  tail call void @abort() #9
  unreachable

48:                                               ; preds = %40
  ret void
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #7 {
  tail call void @testA()
  tail call void @testB()
  tail call void @testC()
  tail call void @testD()
  tail call void @testE()
  tail call void @testF()
  tail call void @testG()
  tail call void @testH()
  tail call void @testI()
  tail call void @testJ()
  tail call void @testK()
  tail call void @testL()
  tail call void @testM()
  tail call void @testN()
  tail call void @testO()
  tail call void @testP()
  tail call void @testQ()
  tail call void @testR()
  tail call void @testS()
  tail call void @testT()
  tail call void @testU()
  tail call void @testV()
  tail call void @testW()
  tail call void @testX()
  tail call void @testY()
  tail call void @testZ()
  tail call void @exit(i32 noundef 0) #9
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #8

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { noreturn nounwind }

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
!10 = !{!8, !8, i64 0}
!11 = !{i64 0, i64 16, !12, i64 16, i64 4, !10}
!12 = !{!13, !13, i64 0}
!13 = !{!"long double", !8, i64 0}
!14 = !{!15, !13, i64 0}
!15 = !{!"W", !13, i64 0, !7, i64 16, !7, i64 17, !7, i64 19}
!16 = !{i64 0, i64 4, !10, i64 4, i64 16, !12}
!17 = !{!18, !13, i64 4}
!18 = !{!"X", !7, i64 0, !7, i64 1, !7, i64 3, !13, i64 4}
!19 = !{!20, !13, i64 4}
!20 = !{!"Y", !7, i64 0, !7, i64 1, !7, i64 2, !13, i64 4}
!21 = !{!22, !13, i64 0}
!22 = !{!"Z", !13, i64 0, !7, i64 16, !7, i64 17, !7, i64 18}
