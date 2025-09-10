; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr51581-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr51581-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global [4096 x i32] zeroinitializer, align 8
@c = dso_local local_unnamed_addr global [4096 x i32] zeroinitializer, align 4
@b = dso_local local_unnamed_addr global [4096 x i32] zeroinitializer, align 4
@d = dso_local local_unnamed_addr global [4096 x i32] zeroinitializer, align 4

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f1() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %11, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 4, !tbaa !6
  %7 = sdiv <4 x i32> %5, splat (i32 3)
  %8 = sdiv <4 x i32> %6, splat (i32 3)
  %9 = getelementptr inbounds nuw i32, ptr @c, i64 %2
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store <4 x i32> %7, ptr %9, align 4, !tbaa !6
  store <4 x i32> %8, ptr %10, align 4, !tbaa !6
  %11 = add nuw i64 %2, 8
  %12 = icmp eq i64 %11, 4096
  br i1 %12, label %13, label %1, !llvm.loop !10

13:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f2() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %11, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @b, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 4, !tbaa !6
  %7 = udiv <4 x i32> %5, splat (i32 3)
  %8 = udiv <4 x i32> %6, splat (i32 3)
  %9 = getelementptr inbounds nuw i32, ptr @d, i64 %2
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store <4 x i32> %7, ptr %9, align 4, !tbaa !6
  store <4 x i32> %8, ptr %10, align 4, !tbaa !6
  %11 = add nuw i64 %2, 8
  %12 = icmp eq i64 %11, 4096
  br i1 %12, label %13, label %1, !llvm.loop !14

13:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f3() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %11, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 4, !tbaa !6
  %7 = sdiv <4 x i32> %5, splat (i32 18)
  %8 = sdiv <4 x i32> %6, splat (i32 18)
  %9 = getelementptr inbounds nuw i32, ptr @c, i64 %2
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store <4 x i32> %7, ptr %9, align 4, !tbaa !6
  store <4 x i32> %8, ptr %10, align 4, !tbaa !6
  %11 = add nuw i64 %2, 8
  %12 = icmp eq i64 %11, 4096
  br i1 %12, label %13, label %1, !llvm.loop !15

13:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f4() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %11, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @b, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 4, !tbaa !6
  %7 = udiv <4 x i32> %5, splat (i32 18)
  %8 = udiv <4 x i32> %6, splat (i32 18)
  %9 = getelementptr inbounds nuw i32, ptr @d, i64 %2
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store <4 x i32> %7, ptr %9, align 4, !tbaa !6
  store <4 x i32> %8, ptr %10, align 4, !tbaa !6
  %11 = add nuw i64 %2, 8
  %12 = icmp eq i64 %11, 4096
  br i1 %12, label %13, label %1, !llvm.loop !16

13:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f5() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %11, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 4, !tbaa !6
  %7 = sdiv <4 x i32> %5, splat (i32 19)
  %8 = sdiv <4 x i32> %6, splat (i32 19)
  %9 = getelementptr inbounds nuw i32, ptr @c, i64 %2
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store <4 x i32> %7, ptr %9, align 4, !tbaa !6
  store <4 x i32> %8, ptr %10, align 4, !tbaa !6
  %11 = add nuw i64 %2, 8
  %12 = icmp eq i64 %11, 4096
  br i1 %12, label %13, label %1, !llvm.loop !17

13:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f6() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %11, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @b, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 4, !tbaa !6
  %7 = udiv <4 x i32> %5, splat (i32 19)
  %8 = udiv <4 x i32> %6, splat (i32 19)
  %9 = getelementptr inbounds nuw i32, ptr @d, i64 %2
  %10 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store <4 x i32> %7, ptr %9, align 4, !tbaa !6
  store <4 x i32> %8, ptr %10, align 4, !tbaa !6
  %11 = add nuw i64 %2, 8
  %12 = icmp eq i64 %11, 4096
  br i1 %12, label %13, label %1, !llvm.loop !18

13:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f7() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %12, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  %4 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %5 = sext <4 x i32> %4 to <4 x i64>
  %6 = mul nsw <4 x i64> %5, splat (i64 1431655766)
  %7 = lshr <4 x i64> %6, splat (i64 32)
  %8 = trunc nuw <4 x i64> %7 to <4 x i32>
  %9 = lshr <4 x i32> %4, splat (i32 31)
  %10 = add <4 x i32> %9, %8
  %11 = getelementptr inbounds nuw i32, ptr @c, i64 %2
  store <4 x i32> %10, ptr %11, align 4, !tbaa !6
  %12 = add nuw i64 %2, 4
  %13 = icmp eq i64 %12, 4096
  br i1 %13, label %14, label %1, !llvm.loop !19

14:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f8() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %17, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @b, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 4, !tbaa !6
  %7 = zext <4 x i32> %5 to <4 x i64>
  %8 = zext <4 x i32> %6 to <4 x i64>
  %9 = mul nuw <4 x i64> %7, splat (i64 2863311531)
  %10 = mul nuw <4 x i64> %8, splat (i64 2863311531)
  %11 = lshr <4 x i64> %9, splat (i64 33)
  %12 = lshr <4 x i64> %10, splat (i64 33)
  %13 = trunc nuw nsw <4 x i64> %11 to <4 x i32>
  %14 = trunc nuw nsw <4 x i64> %12 to <4 x i32>
  %15 = getelementptr inbounds nuw i32, ptr @d, i64 %2
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 16
  store <4 x i32> %13, ptr %15, align 4, !tbaa !6
  store <4 x i32> %14, ptr %16, align 4, !tbaa !6
  %17 = add nuw i64 %2, 8
  %18 = icmp eq i64 %17, 4096
  br i1 %18, label %19, label %1, !llvm.loop !20

19:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f9() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %13, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  %4 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %5 = sext <4 x i32> %4 to <4 x i64>
  %6 = mul nsw <4 x i64> %5, splat (i64 954437177)
  %7 = lshr <4 x i64> %6, splat (i64 32)
  %8 = trunc nuw <4 x i64> %7 to <4 x i32>
  %9 = ashr <4 x i32> %8, splat (i32 2)
  %10 = lshr <4 x i32> %4, splat (i32 31)
  %11 = add nsw <4 x i32> %9, %10
  %12 = getelementptr inbounds nuw i32, ptr @c, i64 %2
  store <4 x i32> %11, ptr %12, align 4, !tbaa !6
  %13 = add nuw i64 %2, 4
  %14 = icmp eq i64 %13, 4096
  br i1 %14, label %15, label %1, !llvm.loop !21

15:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f10() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %17, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @b, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 4, !tbaa !6
  %7 = zext <4 x i32> %5 to <4 x i64>
  %8 = zext <4 x i32> %6 to <4 x i64>
  %9 = mul nuw nsw <4 x i64> %7, splat (i64 954437177)
  %10 = mul nuw nsw <4 x i64> %8, splat (i64 954437177)
  %11 = lshr <4 x i64> %9, splat (i64 34)
  %12 = lshr <4 x i64> %10, splat (i64 34)
  %13 = trunc nuw nsw <4 x i64> %11 to <4 x i32>
  %14 = trunc nuw nsw <4 x i64> %12 to <4 x i32>
  %15 = getelementptr inbounds nuw i32, ptr @d, i64 %2
  %16 = getelementptr inbounds nuw i8, ptr %15, i64 16
  store <4 x i32> %13, ptr %15, align 4, !tbaa !6
  store <4 x i32> %14, ptr %16, align 4, !tbaa !6
  %17 = add nuw i64 %2, 8
  %18 = icmp eq i64 %17, 4096
  br i1 %18, label %19, label %1, !llvm.loop !22

19:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f11() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %13, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  %4 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %5 = sext <4 x i32> %4 to <4 x i64>
  %6 = mul nsw <4 x i64> %5, splat (i64 1808407283)
  %7 = lshr <4 x i64> %6, splat (i64 32)
  %8 = trunc nuw <4 x i64> %7 to <4 x i32>
  %9 = ashr <4 x i32> %8, splat (i32 3)
  %10 = lshr <4 x i32> %4, splat (i32 31)
  %11 = add nsw <4 x i32> %9, %10
  %12 = getelementptr inbounds nuw i32, ptr @c, i64 %2
  store <4 x i32> %11, ptr %12, align 4, !tbaa !6
  %13 = add nuw i64 %2, 4
  %14 = icmp eq i64 %13, 4096
  br i1 %14, label %15, label %1, !llvm.loop !23

15:                                               ; preds = %1
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @f12() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %25, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @b, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %5 = load <4 x i32>, ptr %3, align 4, !tbaa !6
  %6 = load <4 x i32>, ptr %4, align 4, !tbaa !6
  %7 = zext <4 x i32> %5 to <4 x i64>
  %8 = zext <4 x i32> %6 to <4 x i64>
  %9 = mul nuw <4 x i64> %7, splat (i64 2938661835)
  %10 = mul nuw <4 x i64> %8, splat (i64 2938661835)
  %11 = lshr <4 x i64> %9, splat (i64 32)
  %12 = lshr <4 x i64> %10, splat (i64 32)
  %13 = trunc nuw <4 x i64> %11 to <4 x i32>
  %14 = trunc nuw <4 x i64> %12 to <4 x i32>
  %15 = sub <4 x i32> %5, %13
  %16 = sub <4 x i32> %6, %14
  %17 = lshr <4 x i32> %15, splat (i32 1)
  %18 = lshr <4 x i32> %16, splat (i32 1)
  %19 = add <4 x i32> %17, %13
  %20 = add <4 x i32> %18, %14
  %21 = lshr <4 x i32> %19, splat (i32 4)
  %22 = lshr <4 x i32> %20, splat (i32 4)
  %23 = getelementptr inbounds nuw i32, ptr @d, i64 %2
  %24 = getelementptr inbounds nuw i8, ptr %23, i64 16
  store <4 x i32> %21, ptr %23, align 4, !tbaa !6
  store <4 x i32> %22, ptr %24, align 4, !tbaa !6
  %25 = add nuw i64 %2, 8
  %26 = icmp eq i64 %25, 4096
  br i1 %26, label %27, label %1, !llvm.loop !24

27:                                               ; preds = %1
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %0, %1
  %2 = phi i64 [ 0, %0 ], [ %8, %1 ]
  tail call void asm sideeffect "", ""() #3, !srcloc !25
  %3 = getelementptr inbounds nuw i32, ptr @a, i64 %2
  %4 = trunc i64 %2 to i32
  %5 = add i32 %4, -2048
  store i32 %5, ptr %3, align 4, !tbaa !6
  %6 = getelementptr inbounds nuw i32, ptr @b, i64 %2
  %7 = trunc nuw nsw i64 %2 to i32
  store i32 %7, ptr %6, align 4, !tbaa !6
  %8 = add nuw nsw i64 %2, 1
  %9 = icmp eq i64 %8, 4096
  br i1 %9, label %10, label %1, !llvm.loop !26

10:                                               ; preds = %1
  store <2 x i32> <i32 -2147483648, i32 -2147483647>, ptr @a, align 8, !tbaa !6
  store i32 2147483647, ptr getelementptr inbounds nuw (i8, ptr @a, i64 16380), align 4, !tbaa !6
  store i32 -1, ptr getelementptr inbounds nuw (i8, ptr @b, i64 16380), align 4, !tbaa !6
  tail call void @f1()
  tail call void @f2()
  br label %14

11:                                               ; preds = %22
  %12 = add nuw nsw i64 %15, 1
  %13 = icmp eq i64 %12, 4096
  br i1 %13, label %30, label %14, !llvm.loop !27

14:                                               ; preds = %10, %11
  %15 = phi i64 [ 0, %10 ], [ %12, %11 ]
  %16 = getelementptr inbounds nuw i32, ptr @c, i64 %15
  %17 = load i32, ptr %16, align 4, !tbaa !6
  %18 = getelementptr inbounds nuw i32, ptr @a, i64 %15
  %19 = load i32, ptr %18, align 4, !tbaa !6
  %20 = sdiv i32 %19, 3
  %21 = icmp eq i32 %17, %20
  br i1 %21, label %22, label %29

22:                                               ; preds = %14
  %23 = getelementptr inbounds nuw i32, ptr @d, i64 %15
  %24 = load i32, ptr %23, align 4, !tbaa !6
  %25 = getelementptr inbounds nuw i32, ptr @b, i64 %15
  %26 = load i32, ptr %25, align 4, !tbaa !6
  %27 = udiv i32 %26, 3
  %28 = icmp eq i32 %24, %27
  br i1 %28, label %11, label %29

29:                                               ; preds = %22, %14
  tail call void @abort() #4
  unreachable

30:                                               ; preds = %11
  tail call void @f3()
  tail call void @f4()
  br label %34

31:                                               ; preds = %42
  %32 = add nuw nsw i64 %35, 1
  %33 = icmp eq i64 %32, 4096
  br i1 %33, label %50, label %34, !llvm.loop !28

34:                                               ; preds = %30, %31
  %35 = phi i64 [ 0, %30 ], [ %32, %31 ]
  %36 = getelementptr inbounds nuw i32, ptr @c, i64 %35
  %37 = load i32, ptr %36, align 4, !tbaa !6
  %38 = getelementptr inbounds nuw i32, ptr @a, i64 %35
  %39 = load i32, ptr %38, align 4, !tbaa !6
  %40 = sdiv i32 %39, 18
  %41 = icmp eq i32 %37, %40
  br i1 %41, label %42, label %49

42:                                               ; preds = %34
  %43 = getelementptr inbounds nuw i32, ptr @d, i64 %35
  %44 = load i32, ptr %43, align 4, !tbaa !6
  %45 = getelementptr inbounds nuw i32, ptr @b, i64 %35
  %46 = load i32, ptr %45, align 4, !tbaa !6
  %47 = udiv i32 %46, 18
  %48 = icmp eq i32 %44, %47
  br i1 %48, label %31, label %49

49:                                               ; preds = %42, %34
  tail call void @abort() #4
  unreachable

50:                                               ; preds = %31
  tail call void @f5()
  tail call void @f6()
  br label %54

51:                                               ; preds = %62
  %52 = add nuw nsw i64 %55, 1
  %53 = icmp eq i64 %52, 4096
  br i1 %53, label %70, label %54, !llvm.loop !29

54:                                               ; preds = %50, %51
  %55 = phi i64 [ 0, %50 ], [ %52, %51 ]
  %56 = getelementptr inbounds nuw i32, ptr @c, i64 %55
  %57 = load i32, ptr %56, align 4, !tbaa !6
  %58 = getelementptr inbounds nuw i32, ptr @a, i64 %55
  %59 = load i32, ptr %58, align 4, !tbaa !6
  %60 = sdiv i32 %59, 19
  %61 = icmp eq i32 %57, %60
  br i1 %61, label %62, label %69

62:                                               ; preds = %54
  %63 = getelementptr inbounds nuw i32, ptr @d, i64 %55
  %64 = load i32, ptr %63, align 4, !tbaa !6
  %65 = getelementptr inbounds nuw i32, ptr @b, i64 %55
  %66 = load i32, ptr %65, align 4, !tbaa !6
  %67 = udiv i32 %66, 19
  %68 = icmp eq i32 %64, %67
  br i1 %68, label %51, label %69

69:                                               ; preds = %62, %54
  tail call void @abort() #4
  unreachable

70:                                               ; preds = %51
  tail call void @f7()
  tail call void @f8()
  br label %74

71:                                               ; preds = %82
  %72 = add nuw nsw i64 %75, 1
  %73 = icmp eq i64 %72, 4096
  br i1 %73, label %90, label %74, !llvm.loop !30

74:                                               ; preds = %70, %71
  %75 = phi i64 [ 0, %70 ], [ %72, %71 ]
  %76 = getelementptr inbounds nuw i32, ptr @c, i64 %75
  %77 = load i32, ptr %76, align 4, !tbaa !6
  %78 = getelementptr inbounds nuw i32, ptr @a, i64 %75
  %79 = load i32, ptr %78, align 4, !tbaa !6
  %80 = sdiv i32 %79, 3
  %81 = icmp eq i32 %77, %80
  br i1 %81, label %82, label %89

82:                                               ; preds = %74
  %83 = getelementptr inbounds nuw i32, ptr @d, i64 %75
  %84 = load i32, ptr %83, align 4, !tbaa !6
  %85 = getelementptr inbounds nuw i32, ptr @b, i64 %75
  %86 = load i32, ptr %85, align 4, !tbaa !6
  %87 = udiv i32 %86, 3
  %88 = icmp eq i32 %84, %87
  br i1 %88, label %71, label %89

89:                                               ; preds = %82, %74
  tail call void @abort() #4
  unreachable

90:                                               ; preds = %71
  tail call void @f9()
  tail call void @f10()
  br label %94

91:                                               ; preds = %102
  %92 = add nuw nsw i64 %95, 1
  %93 = icmp eq i64 %92, 4096
  br i1 %93, label %110, label %94, !llvm.loop !31

94:                                               ; preds = %90, %91
  %95 = phi i64 [ 0, %90 ], [ %92, %91 ]
  %96 = getelementptr inbounds nuw i32, ptr @c, i64 %95
  %97 = load i32, ptr %96, align 4, !tbaa !6
  %98 = getelementptr inbounds nuw i32, ptr @a, i64 %95
  %99 = load i32, ptr %98, align 4, !tbaa !6
  %100 = sdiv i32 %99, 18
  %101 = icmp eq i32 %97, %100
  br i1 %101, label %102, label %109

102:                                              ; preds = %94
  %103 = getelementptr inbounds nuw i32, ptr @d, i64 %95
  %104 = load i32, ptr %103, align 4, !tbaa !6
  %105 = getelementptr inbounds nuw i32, ptr @b, i64 %95
  %106 = load i32, ptr %105, align 4, !tbaa !6
  %107 = udiv i32 %106, 18
  %108 = icmp eq i32 %104, %107
  br i1 %108, label %91, label %109

109:                                              ; preds = %102, %94
  tail call void @abort() #4
  unreachable

110:                                              ; preds = %91
  tail call void @f11()
  tail call void @f12()
  br label %114

111:                                              ; preds = %122
  %112 = add nuw nsw i64 %115, 1
  %113 = icmp eq i64 %112, 4096
  br i1 %113, label %130, label %114, !llvm.loop !32

114:                                              ; preds = %110, %111
  %115 = phi i64 [ 0, %110 ], [ %112, %111 ]
  %116 = getelementptr inbounds nuw i32, ptr @c, i64 %115
  %117 = load i32, ptr %116, align 4, !tbaa !6
  %118 = getelementptr inbounds nuw i32, ptr @a, i64 %115
  %119 = load i32, ptr %118, align 4, !tbaa !6
  %120 = sdiv i32 %119, 19
  %121 = icmp eq i32 %117, %120
  br i1 %121, label %122, label %129

122:                                              ; preds = %114
  %123 = getelementptr inbounds nuw i32, ptr @d, i64 %115
  %124 = load i32, ptr %123, align 4, !tbaa !6
  %125 = getelementptr inbounds nuw i32, ptr @b, i64 %115
  %126 = load i32, ptr %125, align 4, !tbaa !6
  %127 = udiv i32 %126, 19
  %128 = icmp eq i32 %124, %127
  br i1 %128, label %111, label %129

129:                                              ; preds = %122, %114
  tail call void @abort() #4
  unreachable

130:                                              ; preds = %111
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nounwind }
attributes #4 = { noreturn nounwind }

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
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !12, !13}
!15 = distinct !{!15, !11, !12, !13}
!16 = distinct !{!16, !11, !12, !13}
!17 = distinct !{!17, !11, !12, !13}
!18 = distinct !{!18, !11, !12, !13}
!19 = distinct !{!19, !11, !12, !13}
!20 = distinct !{!20, !11, !12, !13}
!21 = distinct !{!21, !11, !12, !13}
!22 = distinct !{!22, !11, !12, !13}
!23 = distinct !{!23, !11, !12, !13}
!24 = distinct !{!24, !11, !12, !13}
!25 = !{i64 2020}
!26 = distinct !{!26, !11}
!27 = distinct !{!27, !11}
!28 = distinct !{!28, !11}
!29 = distinct !{!29, !11}
!30 = distinct !{!30, !11}
!31 = distinct !{!31, !11}
!32 = distinct !{!32, !11}
