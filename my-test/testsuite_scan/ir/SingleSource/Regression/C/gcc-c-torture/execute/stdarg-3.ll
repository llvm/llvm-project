; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stdarg-3.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/stdarg-3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S1 = type { i32, double, i32, double }
%struct.S2 = type { double, i64 }
%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@bar_arg = dso_local local_unnamed_addr global i32 0, align 4
@x = dso_local local_unnamed_addr global i64 0, align 8
@d = dso_local local_unnamed_addr global double 0.000000e+00, align 8
@s1 = dso_local local_unnamed_addr global %struct.S1 zeroinitializer, align 8
@s2 = dso_local local_unnamed_addr global %struct.S2 zeroinitializer, align 8
@y = dso_local local_unnamed_addr global i32 0, align 4
@foo_arg = dso_local local_unnamed_addr global i32 0, align 4
@gap = dso_local local_unnamed_addr global %struct.__va_list zeroinitializer, align 8

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @bar(i32 noundef %0) local_unnamed_addr #0 {
  store i32 %0, ptr @bar_arg, align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @f1(i32 noundef %0, ...) local_unnamed_addr #1 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %31

4:                                                ; preds = %1
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %8 = load ptr, ptr %7, align 8
  %9 = load i32, ptr %6, align 8
  br label %10

10:                                               ; preds = %4, %25
  %11 = phi i32 [ %9, %4 ], [ %26, %25 ]
  %12 = phi i32 [ %0, %4 ], [ %14, %25 ]
  %13 = phi ptr [ %5, %4 ], [ %27, %25 ]
  %14 = add nsw i32 %12, -1
  %15 = icmp sgt i32 %11, -1
  br i1 %15, label %22, label %16

16:                                               ; preds = %10
  %17 = add nsw i32 %11, 8
  store i32 %17, ptr %6, align 8
  %18 = icmp samesign ult i32 %11, -7
  br i1 %18, label %19, label %22

19:                                               ; preds = %16
  %20 = sext i32 %11 to i64
  %21 = getelementptr inbounds i8, ptr %8, i64 %20
  br label %25

22:                                               ; preds = %16, %10
  %23 = phi i32 [ %17, %16 ], [ %11, %10 ]
  %24 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store ptr %24, ptr %2, align 8
  br label %25

25:                                               ; preds = %22, %19
  %26 = phi i32 [ %17, %19 ], [ %23, %22 ]
  %27 = phi ptr [ %13, %19 ], [ %24, %22 ]
  %28 = phi ptr [ %21, %19 ], [ %13, %22 ]
  %29 = load i64, ptr %28, align 8, !tbaa !10
  store i64 %29, ptr @x, align 8, !tbaa !10
  %30 = icmp samesign ugt i32 %12, 1
  br i1 %30, label %10, label %31, !llvm.loop !12

31:                                               ; preds = %25, %1
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @f2(i32 noundef %0, ...) local_unnamed_addr #1 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %31

4:                                                ; preds = %1
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %8 = load ptr, ptr %7, align 8
  %9 = load i32, ptr %6, align 4
  br label %10

10:                                               ; preds = %4, %25
  %11 = phi i32 [ %9, %4 ], [ %26, %25 ]
  %12 = phi i32 [ %0, %4 ], [ %14, %25 ]
  %13 = phi ptr [ %5, %4 ], [ %27, %25 ]
  %14 = add nsw i32 %12, -1
  %15 = icmp sgt i32 %11, -1
  br i1 %15, label %22, label %16

16:                                               ; preds = %10
  %17 = add nsw i32 %11, 16
  store i32 %17, ptr %6, align 4
  %18 = icmp samesign ult i32 %11, -15
  br i1 %18, label %19, label %22

19:                                               ; preds = %16
  %20 = sext i32 %11 to i64
  %21 = getelementptr inbounds i8, ptr %8, i64 %20
  br label %25

22:                                               ; preds = %16, %10
  %23 = phi i32 [ %17, %16 ], [ %11, %10 ]
  %24 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store ptr %24, ptr %2, align 8
  br label %25

25:                                               ; preds = %22, %19
  %26 = phi i32 [ %17, %19 ], [ %23, %22 ]
  %27 = phi ptr [ %13, %19 ], [ %24, %22 ]
  %28 = phi ptr [ %21, %19 ], [ %13, %22 ]
  %29 = load double, ptr %28, align 8, !tbaa !14
  store double %29, ptr @d, align 8, !tbaa !14
  %30 = icmp samesign ugt i32 %12, 1
  br i1 %30, label %10, label %31, !llvm.loop !16

31:                                               ; preds = %25, %1
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @f3(i32 noundef %0, ...) local_unnamed_addr #1 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %28

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 8
  br label %7

7:                                                ; preds = %4, %22
  %8 = phi i32 [ %0, %4 ], [ %9, %22 ]
  %9 = add nsw i32 %8, -1
  call void @llvm.va_start.p0(ptr nonnull %2)
  %10 = load i32, ptr %5, align 8
  %11 = icmp sgt i32 %10, -1
  br i1 %11, label %19, label %12

12:                                               ; preds = %7
  %13 = add nsw i32 %10, 8
  store i32 %13, ptr %5, align 8
  %14 = icmp samesign ult i32 %10, -7
  br i1 %14, label %15, label %19

15:                                               ; preds = %12
  %16 = load ptr, ptr %6, align 8
  %17 = sext i32 %10 to i64
  %18 = getelementptr inbounds i8, ptr %16, i64 %17
  br label %22

19:                                               ; preds = %12, %7
  %20 = load ptr, ptr %2, align 8
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 8
  store ptr %21, ptr %2, align 8
  br label %22

22:                                               ; preds = %19, %15
  %23 = phi ptr [ %18, %15 ], [ %20, %19 ]
  %24 = load i64, ptr %23, align 8, !tbaa !10
  store i64 %24, ptr @x, align 8, !tbaa !10
  call void @llvm.va_end.p0(ptr nonnull %2)
  %25 = load i64, ptr @x, align 8, !tbaa !10
  %26 = trunc i64 %25 to i32
  store i32 %26, ptr @bar_arg, align 4, !tbaa !6
  %27 = icmp samesign ugt i32 %8, 1
  br i1 %27, label %7, label %28, !llvm.loop !17

28:                                               ; preds = %22, %1
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @f4(i32 noundef %0, ...) local_unnamed_addr #1 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %29

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 28
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 16
  br label %7

7:                                                ; preds = %4, %22
  %8 = phi i32 [ %0, %4 ], [ %9, %22 ]
  %9 = add nsw i32 %8, -1
  call void @llvm.va_start.p0(ptr nonnull %2)
  %10 = load i32, ptr %5, align 4
  %11 = icmp sgt i32 %10, -1
  br i1 %11, label %19, label %12

12:                                               ; preds = %7
  %13 = add nsw i32 %10, 16
  store i32 %13, ptr %5, align 4
  %14 = icmp samesign ult i32 %10, -15
  br i1 %14, label %15, label %19

15:                                               ; preds = %12
  %16 = load ptr, ptr %6, align 8
  %17 = sext i32 %10 to i64
  %18 = getelementptr inbounds i8, ptr %16, i64 %17
  br label %22

19:                                               ; preds = %12, %7
  %20 = load ptr, ptr %2, align 8
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 8
  store ptr %21, ptr %2, align 8
  br label %22

22:                                               ; preds = %19, %15
  %23 = phi ptr [ %18, %15 ], [ %20, %19 ]
  %24 = load double, ptr %23, align 8, !tbaa !14
  store double %24, ptr @d, align 8, !tbaa !14
  call void @llvm.va_end.p0(ptr nonnull %2)
  %25 = load double, ptr @d, align 8, !tbaa !14
  %26 = fadd double %25, 4.000000e+00
  %27 = fptosi double %26 to i32
  store i32 %27, ptr @bar_arg, align 4, !tbaa !6
  %28 = icmp samesign ugt i32 %8, 1
  br i1 %28, label %7, label %29, !llvm.loop !18

29:                                               ; preds = %22, %1
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @f5(i32 noundef %0, ...) local_unnamed_addr #1 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %31

4:                                                ; preds = %1
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %8 = load ptr, ptr %7, align 8
  %9 = load i32, ptr %6, align 8
  br label %10

10:                                               ; preds = %4, %25
  %11 = phi i32 [ %9, %4 ], [ %26, %25 ]
  %12 = phi i32 [ %0, %4 ], [ %14, %25 ]
  %13 = phi ptr [ %5, %4 ], [ %27, %25 ]
  %14 = add nsw i32 %12, -1
  %15 = icmp sgt i32 %11, -1
  br i1 %15, label %22, label %16

16:                                               ; preds = %10
  %17 = add nsw i32 %11, 8
  store i32 %17, ptr %6, align 8
  %18 = icmp samesign ult i32 %11, -7
  br i1 %18, label %19, label %22

19:                                               ; preds = %16
  %20 = sext i32 %11 to i64
  %21 = getelementptr inbounds i8, ptr %8, i64 %20
  br label %25

22:                                               ; preds = %16, %10
  %23 = phi i32 [ %17, %16 ], [ %11, %10 ]
  %24 = getelementptr inbounds nuw i8, ptr %13, i64 8
  store ptr %24, ptr %2, align 8
  br label %25

25:                                               ; preds = %22, %19
  %26 = phi i32 [ %17, %19 ], [ %23, %22 ]
  %27 = phi ptr [ %13, %19 ], [ %24, %22 ]
  %28 = phi ptr [ %21, %19 ], [ %13, %22 ]
  %29 = load ptr, ptr %28, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) @s1, ptr noundef nonnull align 8 dereferenceable(32) %29, i64 32, i1 false), !tbaa.struct !19
  %30 = icmp samesign ugt i32 %12, 1
  br i1 %30, label %10, label %31, !llvm.loop !20

31:                                               ; preds = %25, %1
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #4

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @f6(i32 noundef %0, ...) local_unnamed_addr #1 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %30

4:                                                ; preds = %1
  %5 = load ptr, ptr %2, align 8
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %8 = load ptr, ptr %7, align 8
  %9 = load i32, ptr %6, align 8
  br label %10

10:                                               ; preds = %4, %25
  %11 = phi i32 [ %9, %4 ], [ %26, %25 ]
  %12 = phi i32 [ %0, %4 ], [ %14, %25 ]
  %13 = phi ptr [ %5, %4 ], [ %27, %25 ]
  %14 = add nsw i32 %12, -1
  %15 = icmp sgt i32 %11, -1
  br i1 %15, label %22, label %16

16:                                               ; preds = %10
  %17 = add nsw i32 %11, 16
  store i32 %17, ptr %6, align 8
  %18 = icmp samesign ult i32 %11, -15
  br i1 %18, label %19, label %22

19:                                               ; preds = %16
  %20 = sext i32 %11 to i64
  %21 = getelementptr inbounds i8, ptr %8, i64 %20
  br label %25

22:                                               ; preds = %16, %10
  %23 = phi i32 [ %17, %16 ], [ %11, %10 ]
  %24 = getelementptr inbounds nuw i8, ptr %13, i64 16
  store ptr %24, ptr %2, align 8
  br label %25

25:                                               ; preds = %22, %19
  %26 = phi i32 [ %17, %19 ], [ %23, %22 ]
  %27 = phi ptr [ %13, %19 ], [ %24, %22 ]
  %28 = phi ptr [ %21, %19 ], [ %13, %22 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) @s2, ptr noundef nonnull align 8 dereferenceable(16) %28, i64 16, i1 false), !tbaa.struct !21
  %29 = icmp samesign ugt i32 %12, 1
  br i1 %29, label %10, label %30, !llvm.loop !22

30:                                               ; preds = %25, %1
  call void @llvm.va_end.p0(ptr nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @f7(i32 noundef %0, ...) local_unnamed_addr #1 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %27

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 8
  br label %7

7:                                                ; preds = %4, %22
  %8 = phi i32 [ %0, %4 ], [ %9, %22 ]
  %9 = add nsw i32 %8, -1
  call void @llvm.va_start.p0(ptr nonnull %2)
  %10 = load i32, ptr %5, align 8
  %11 = icmp sgt i32 %10, -1
  br i1 %11, label %19, label %12

12:                                               ; preds = %7
  %13 = add nsw i32 %10, 8
  store i32 %13, ptr %5, align 8
  %14 = icmp samesign ult i32 %10, -7
  br i1 %14, label %15, label %19

15:                                               ; preds = %12
  %16 = load ptr, ptr %6, align 8
  %17 = sext i32 %10 to i64
  %18 = getelementptr inbounds i8, ptr %16, i64 %17
  br label %22

19:                                               ; preds = %12, %7
  %20 = load ptr, ptr %2, align 8
  %21 = getelementptr inbounds nuw i8, ptr %20, i64 8
  store ptr %21, ptr %2, align 8
  br label %22

22:                                               ; preds = %19, %15
  %23 = phi ptr [ %18, %15 ], [ %20, %19 ]
  %24 = load ptr, ptr %23, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(32) @s1, ptr noundef nonnull align 8 dereferenceable(32) %24, i64 32, i1 false), !tbaa.struct !19
  call void @llvm.va_end.p0(ptr nonnull %2)
  %25 = load i32, ptr @s1, align 8, !tbaa !23
  store i32 %25, ptr @bar_arg, align 4, !tbaa !6
  %26 = icmp samesign ugt i32 %8, 1
  br i1 %26, label %7, label %27, !llvm.loop !25

27:                                               ; preds = %22, %1
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind uwtable
define dso_local void @f8(i32 noundef %0, ...) local_unnamed_addr #1 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  %3 = icmp sgt i32 %0, 0
  br i1 %3, label %4, label %39

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 8
  br label %7

7:                                                ; preds = %4, %33
  %8 = phi i32 [ %0, %4 ], [ %9, %33 ]
  %9 = add nsw i32 %8, -1
  call void @llvm.va_start.p0(ptr nonnull %2)
  %10 = load i32, ptr %5, align 8
  %11 = icmp sgt i32 %10, -1
  br i1 %11, label %15, label %12

12:                                               ; preds = %7
  %13 = add nsw i32 %10, 16
  store i32 %13, ptr %5, align 8
  %14 = icmp samesign ult i32 %10, -15
  br i1 %14, label %18, label %15

15:                                               ; preds = %7, %12
  %16 = load ptr, ptr %2, align 8
  %17 = getelementptr inbounds nuw i8, ptr %16, i64 16
  store ptr %17, ptr %2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) @s2, ptr noundef nonnull align 8 dereferenceable(16) %16, i64 16, i1 false), !tbaa.struct !21
  br label %30

18:                                               ; preds = %12
  %19 = load ptr, ptr %6, align 8
  %20 = sext i32 %10 to i64
  %21 = getelementptr inbounds i8, ptr %19, i64 %20
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) @s2, ptr noundef nonnull align 8 dereferenceable(16) %21, i64 16, i1 false), !tbaa.struct !21
  %22 = icmp sgt i32 %10, -17
  br i1 %22, label %30, label %23

23:                                               ; preds = %18
  %24 = add nsw i32 %10, 24
  store i32 %24, ptr %5, align 8
  %25 = icmp samesign ult i32 %13, -7
  br i1 %25, label %26, label %30

26:                                               ; preds = %23
  %27 = load ptr, ptr %6, align 8
  %28 = sext i32 %13 to i64
  %29 = getelementptr inbounds i8, ptr %27, i64 %28
  br label %33

30:                                               ; preds = %15, %23, %18
  %31 = load ptr, ptr %2, align 8
  %32 = getelementptr inbounds nuw i8, ptr %31, i64 8
  store ptr %32, ptr %2, align 8
  br label %33

33:                                               ; preds = %30, %26
  %34 = phi ptr [ %29, %26 ], [ %31, %30 ]
  %35 = load i32, ptr %34, align 8, !tbaa !6
  store i32 %35, ptr @y, align 4, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %2)
  %36 = load i64, ptr getelementptr inbounds nuw (i8, ptr @s2, i64 8), align 8, !tbaa !26
  %37 = trunc i64 %36 to i32
  store i32 %37, ptr @bar_arg, align 4, !tbaa !6
  %38 = icmp samesign ugt i32 %8, 1
  br i1 %38, label %7, label %39, !llvm.loop !28

39:                                               ; preds = %33, %1
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = alloca %struct.S1, align 8
  %2 = alloca %struct.S1, align 8
  %3 = alloca %struct.S1, align 8
  %4 = alloca %struct.S1, align 8
  %5 = alloca %struct.S1, align 8
  %6 = alloca %struct.S1, align 8
  %7 = alloca %struct.S1, align 8
  %8 = alloca %struct.S1, align 8
  %9 = alloca %struct.S1, align 8
  tail call void (i32, ...) @f1(i32 noundef 7, i64 noundef 1, i64 noundef 2, i64 noundef 3, i64 noundef 5, i64 noundef 7, i64 noundef 9, i64 noundef 11, i64 noundef 13)
  %10 = load i64, ptr @x, align 8, !tbaa !10
  %11 = icmp eq i64 %10, 11
  br i1 %11, label %13, label %12

12:                                               ; preds = %0
  tail call void @abort() #8
  unreachable

13:                                               ; preds = %0
  tail call void (i32, ...) @f2(i32 noundef 6, double noundef 1.000000e+00, double noundef 2.000000e+00, double noundef 4.000000e+00, double noundef 8.000000e+00, double noundef 1.600000e+01, double noundef 3.200000e+01, double noundef 6.400000e+01)
  %14 = load double, ptr @d, align 8, !tbaa !14
  %15 = fcmp une double %14, 3.200000e+01
  br i1 %15, label %16, label %17

16:                                               ; preds = %13
  tail call void @abort() #8
  unreachable

17:                                               ; preds = %13
  tail call void (i32, ...) @f3(i32 noundef 2, i64 noundef 1, i64 noundef 3)
  %18 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %19 = icmp ne i32 %18, 1
  %20 = load i64, ptr @x, align 8
  %21 = icmp ne i64 %20, 1
  %22 = select i1 %19, i1 true, i1 %21
  br i1 %22, label %23, label %24

23:                                               ; preds = %17
  tail call void @abort() #8
  unreachable

24:                                               ; preds = %17
  tail call void (i32, ...) @f4(i32 noundef 2, double noundef 1.700000e+01, double noundef 1.900000e+01)
  %25 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %26 = icmp ne i32 %25, 21
  %27 = load double, ptr @d, align 8
  %28 = fcmp une double %27, 1.700000e+01
  %29 = select i1 %26, i1 true, i1 %28
  br i1 %29, label %30, label %31

30:                                               ; preds = %24
  tail call void @abort() #8
  unreachable

31:                                               ; preds = %24
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #7
  store i32 131, ptr %1, align 8, !tbaa !6
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store double 1.500000e+01, ptr %32, align 8, !tbaa !14
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 16
  store i32 251, ptr %33, align 8, !tbaa !6
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 24
  store double 1.910000e+02, ptr %34, align 8, !tbaa !14
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  store i32 131, ptr %2, align 8, !tbaa !6
  %35 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store double 1.500000e+01, ptr %35, align 8, !tbaa !14
  %36 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store i32 254, ptr %36, align 8, !tbaa !6
  %37 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store double 1.780000e+02, ptr %37, align 8, !tbaa !14
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #7
  store i32 131, ptr %3, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store double 1.500000e+01, ptr %38, align 8, !tbaa !14
  %39 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store i32 251, ptr %39, align 8, !tbaa !6
  %40 = getelementptr inbounds nuw i8, ptr %3, i64 24
  store double 1.910000e+02, ptr %40, align 8, !tbaa !14
  call void (i32, ...) @f5(i32 noundef 2, ptr dead_on_return noundef nonnull %1, ptr dead_on_return noundef nonnull %2, ptr dead_on_return noundef nonnull %3)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #7
  %41 = load i32, ptr @s1, align 8, !tbaa !23
  %42 = icmp ne i32 %41, 131
  %43 = load i32, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 16), align 8
  %44 = icmp ne i32 %43, 254
  %45 = select i1 %42, i1 true, i1 %44
  %46 = load double, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 8), align 8
  %47 = fcmp une double %46, 1.500000e+01
  %48 = select i1 %45, i1 true, i1 %47
  %49 = load double, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 24), align 8
  %50 = fcmp une double %49, 1.780000e+02
  %51 = select i1 %48, i1 true, i1 %50
  br i1 %51, label %52, label %53

52:                                               ; preds = %31
  call void @abort() #8
  unreachable

53:                                               ; preds = %31
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #7
  store i32 131, ptr %4, align 8, !tbaa !6
  %54 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store double 1.500000e+01, ptr %54, align 8, !tbaa !14
  %55 = getelementptr inbounds nuw i8, ptr %4, i64 16
  store i32 251, ptr %55, align 8, !tbaa !6
  %56 = getelementptr inbounds nuw i8, ptr %4, i64 24
  store double 1.910000e+02, ptr %56, align 8, !tbaa !14
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #7
  store i32 131, ptr %5, align 8, !tbaa !6
  %57 = getelementptr inbounds nuw i8, ptr %5, i64 8
  store double 1.500000e+01, ptr %57, align 8, !tbaa !14
  %58 = getelementptr inbounds nuw i8, ptr %5, i64 16
  store i32 254, ptr %58, align 8, !tbaa !6
  %59 = getelementptr inbounds nuw i8, ptr %5, i64 24
  store double 1.780000e+02, ptr %59, align 8, !tbaa !14
  call void @llvm.lifetime.start.p0(ptr nonnull %6) #7
  store i32 131, ptr %6, align 8, !tbaa !6
  %60 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store double 1.500000e+01, ptr %60, align 8, !tbaa !14
  %61 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store i32 251, ptr %61, align 8, !tbaa !6
  %62 = getelementptr inbounds nuw i8, ptr %6, i64 24
  store double 1.910000e+02, ptr %62, align 8, !tbaa !14
  call void (i32, ...) @f5(i32 noundef 3, ptr dead_on_return noundef nonnull %4, ptr dead_on_return noundef nonnull %5, ptr dead_on_return noundef nonnull %6)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %6) #7
  %63 = load i32, ptr @s1, align 8, !tbaa !23
  %64 = icmp ne i32 %63, 131
  %65 = load i32, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 16), align 8
  %66 = icmp ne i32 %65, 251
  %67 = select i1 %64, i1 true, i1 %66
  %68 = load double, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 8), align 8
  %69 = fcmp une double %68, 1.500000e+01
  %70 = select i1 %67, i1 true, i1 %69
  %71 = load double, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 24), align 8
  %72 = fcmp une double %71, 1.910000e+02
  %73 = select i1 %70, i1 true, i1 %72
  br i1 %73, label %74, label %75

74:                                               ; preds = %53
  call void @abort() #8
  unreachable

75:                                               ; preds = %53
  call void (i32, ...) @f6(i32 noundef 2, [2 x i64] [i64 4625196817309499392, i64 138], [2 x i64] [i64 4640396466051874816, i64 257], [2 x i64] [i64 4625196817309499392, i64 138])
  %76 = load i64, ptr getelementptr inbounds nuw (i8, ptr @s2, i64 8), align 8, !tbaa !26
  %77 = icmp ne i64 %76, 257
  %78 = load double, ptr @s2, align 8
  %79 = fcmp une double %78, 1.760000e+02
  %80 = select i1 %77, i1 true, i1 %79
  br i1 %80, label %81, label %82

81:                                               ; preds = %75
  call void @abort() #8
  unreachable

82:                                               ; preds = %75
  call void (i32, ...) @f6(i32 noundef 3, [2 x i64] [i64 4625196817309499392, i64 138], [2 x i64] [i64 4640396466051874816, i64 257], [2 x i64] [i64 4625196817309499392, i64 138])
  %83 = load i64, ptr getelementptr inbounds nuw (i8, ptr @s2, i64 8), align 8, !tbaa !26
  %84 = icmp ne i64 %83, 138
  %85 = load double, ptr @s2, align 8
  %86 = fcmp une double %85, 1.600000e+01
  %87 = select i1 %84, i1 true, i1 %86
  br i1 %87, label %88, label %89

88:                                               ; preds = %82
  call void @abort() #8
  unreachable

89:                                               ; preds = %82
  call void @llvm.lifetime.start.p0(ptr nonnull %7) #7
  store i32 131, ptr %7, align 8, !tbaa !6
  %90 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store double 1.500000e+01, ptr %90, align 8, !tbaa !14
  %91 = getelementptr inbounds nuw i8, ptr %7, i64 16
  store i32 254, ptr %91, align 8, !tbaa !6
  %92 = getelementptr inbounds nuw i8, ptr %7, i64 24
  store double 1.780000e+02, ptr %92, align 8, !tbaa !14
  call void @llvm.lifetime.start.p0(ptr nonnull %8) #7
  store i32 131, ptr %8, align 8, !tbaa !6
  %93 = getelementptr inbounds nuw i8, ptr %8, i64 8
  store double 1.500000e+01, ptr %93, align 8, !tbaa !14
  %94 = getelementptr inbounds nuw i8, ptr %8, i64 16
  store i32 251, ptr %94, align 8, !tbaa !6
  %95 = getelementptr inbounds nuw i8, ptr %8, i64 24
  store double 1.910000e+02, ptr %95, align 8, !tbaa !14
  call void @llvm.lifetime.start.p0(ptr nonnull %9) #7
  store i32 131, ptr %9, align 8, !tbaa !6
  %96 = getelementptr inbounds nuw i8, ptr %9, i64 8
  store double 1.500000e+01, ptr %96, align 8, !tbaa !14
  %97 = getelementptr inbounds nuw i8, ptr %9, i64 16
  store i32 251, ptr %97, align 8, !tbaa !6
  %98 = getelementptr inbounds nuw i8, ptr %9, i64 24
  store double 1.910000e+02, ptr %98, align 8, !tbaa !14
  call void (i32, ...) @f7(i32 noundef 2, ptr dead_on_return noundef nonnull %7, ptr dead_on_return noundef nonnull %8, ptr dead_on_return noundef nonnull %9)
  call void @llvm.lifetime.end.p0(ptr nonnull %7) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %8) #7
  call void @llvm.lifetime.end.p0(ptr nonnull %9) #7
  %99 = load i32, ptr @s1, align 8, !tbaa !23
  %100 = icmp ne i32 %99, 131
  %101 = load i32, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 16), align 8
  %102 = icmp ne i32 %101, 254
  %103 = select i1 %100, i1 true, i1 %102
  %104 = load double, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 8), align 8
  %105 = fcmp une double %104, 1.500000e+01
  %106 = select i1 %103, i1 true, i1 %105
  %107 = load double, ptr getelementptr inbounds nuw (i8, ptr @s1, i64 24), align 8
  %108 = fcmp une double %107, 1.780000e+02
  %109 = select i1 %106, i1 true, i1 %108
  br i1 %109, label %110, label %111

110:                                              ; preds = %89
  call void @abort() #8
  unreachable

111:                                              ; preds = %89
  %112 = load i32, ptr @bar_arg, align 4, !tbaa !6
  %113 = icmp eq i32 %112, 131
  br i1 %113, label %115, label %114

114:                                              ; preds = %111
  call void @abort() #8
  unreachable

115:                                              ; preds = %111
  call void (i32, ...) @f8(i32 noundef 3, [2 x i64] [i64 4640396466051874816, i64 257], [2 x i64] [i64 4625196817309499392, i64 138], [2 x i64] [i64 4625196817309499392, i64 138])
  %116 = load i64, ptr getelementptr inbounds nuw (i8, ptr @s2, i64 8), align 8, !tbaa !26
  %117 = icmp ne i64 %116, 257
  %118 = load double, ptr @s2, align 8
  %119 = fcmp une double %118, 1.760000e+02
  %120 = select i1 %117, i1 true, i1 %119
  br i1 %120, label %121, label %122

121:                                              ; preds = %115
  call void @abort() #8
  unreachable

122:                                              ; preds = %115
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #6

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"long", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!15, !15, i64 0}
!15 = !{!"double", !8, i64 0}
!16 = distinct !{!16, !13}
!17 = distinct !{!17, !13}
!18 = distinct !{!18, !13}
!19 = !{i64 0, i64 4, !6, i64 8, i64 8, !14, i64 16, i64 4, !6, i64 24, i64 8, !14}
!20 = distinct !{!20, !13}
!21 = !{i64 0, i64 8, !14, i64 8, i64 8, !10}
!22 = distinct !{!22, !13}
!23 = !{!24, !7, i64 0}
!24 = !{!"S1", !7, i64 0, !15, i64 8, !7, i64 16, !15, i64 24}
!25 = distinct !{!25, !13}
!26 = !{!27, !11, i64 8}
!27 = !{!"S2", !15, i64 0, !11, i64 8}
!28 = distinct !{!28, !13}
