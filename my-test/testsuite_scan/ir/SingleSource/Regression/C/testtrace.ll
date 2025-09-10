; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/testtrace.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/testtrace.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.DummyStruct = type { ptr, i32 }

@.str = private unnamed_addr constant [28 x i8] c"&S1 = %p\09&S2 = %p\09&S3 = %p\0A\00", align 1
@testAllocaOrder.count = internal unnamed_addr global i32 0, align 4
@.str.1 = private unnamed_addr constant [10 x i8] c"sum = %d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @AddCounts(ptr noundef %0, ptr noundef %1, ptr noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp eq i32 %3, 0
  br i1 %5, label %6, label %8

6:                                                ; preds = %4
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef %0, ptr noundef %1, ptr noundef %2)
  br label %8

8:                                                ; preds = %6, %4
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load i32, ptr %9, align 8, !tbaa !6
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = load i32, ptr %11, align 8, !tbaa !6
  %13 = add nsw i32 %12, %10
  %14 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %15 = load i32, ptr %14, align 8, !tbaa !6
  %16 = add nsw i32 %13, %15
  ret i32 %16
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @testAllocaOrder(i32 noundef %0) local_unnamed_addr #0 {
  %2 = alloca %struct.DummyStruct, align 8
  %3 = alloca %struct.DummyStruct, align 8
  %4 = alloca %struct.DummyStruct, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  %5 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %6 = add nsw i32 %5, 1
  %7 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i32 %6, ptr %7, align 8, !tbaa !6
  %8 = add nsw i32 %5, 2
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store i32 %8, ptr %9, align 8, !tbaa !6
  %10 = add nsw i32 %5, 3
  store i32 %10, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %11 = getelementptr inbounds nuw i8, ptr %4, i64 8
  store i32 %10, ptr %11, align 8, !tbaa !6
  %12 = icmp eq i32 %0, 0
  br i1 %12, label %13, label %18

13:                                               ; preds = %1
  %14 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %2, ptr noundef nonnull %3, ptr noundef nonnull %4)
  %15 = load i32, ptr %7, align 8, !tbaa !6
  %16 = load i32, ptr %9, align 8, !tbaa !6
  %17 = load i32, ptr %11, align 8, !tbaa !6
  br label %18

18:                                               ; preds = %1, %13
  %19 = phi i32 [ %10, %1 ], [ %17, %13 ]
  %20 = phi i32 [ %8, %1 ], [ %16, %13 ]
  %21 = phi i32 [ %6, %1 ], [ %15, %13 ]
  %22 = add nsw i32 %20, %21
  %23 = add nsw i32 %22, %19
  %24 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %23)
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #3
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = alloca %struct.DummyStruct, align 8
  %4 = alloca %struct.DummyStruct, align 8
  %5 = alloca %struct.DummyStruct, align 8
  %6 = icmp sgt i32 %0, 1
  br i1 %6, label %7, label %16

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !14
  %10 = load i8, ptr %9, align 1
  %11 = icmp eq i8 %10, 45
  br i1 %11, label %12, label %16

12:                                               ; preds = %7
  %13 = getelementptr inbounds nuw i8, ptr %9, i64 1
  %14 = load i8, ptr %13, align 1
  %15 = icmp eq i8 %14, 100
  br i1 %15, label %20, label %16

16:                                               ; preds = %2, %7, %12
  %17 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %18 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %19 = getelementptr inbounds nuw i8, ptr %5, i64 8
  br label %27

20:                                               ; preds = %12
  %21 = getelementptr inbounds nuw i8, ptr %9, i64 2
  %22 = load i8, ptr %21, align 1
  %23 = icmp eq i8 %22, 0
  %24 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %25 = getelementptr inbounds nuw i8, ptr %4, i64 8
  %26 = getelementptr inbounds nuw i8, ptr %5, i64 8
  br i1 %23, label %100, label %27

27:                                               ; preds = %16, %20
  %28 = phi ptr [ %19, %16 ], [ %26, %20 ]
  %29 = phi ptr [ %18, %16 ], [ %25, %20 ]
  %30 = phi ptr [ %17, %16 ], [ %24, %20 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %31 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %32 = add nsw i32 %31, 1
  store i32 %32, ptr %30, align 8, !tbaa !6
  %33 = add nsw i32 %31, 2
  store i32 %33, ptr %29, align 8, !tbaa !6
  %34 = add nsw i32 %31, 3
  store i32 %34, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %34, ptr %28, align 8, !tbaa !6
  %35 = add i32 %33, %34
  %36 = add i32 %35, %32
  %37 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %36)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %38 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %39 = add nsw i32 %38, 1
  store i32 %39, ptr %30, align 8, !tbaa !6
  %40 = add nsw i32 %38, 2
  store i32 %40, ptr %29, align 8, !tbaa !6
  %41 = add nsw i32 %38, 3
  store i32 %41, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %41, ptr %28, align 8, !tbaa !6
  %42 = add i32 %40, %41
  %43 = add i32 %42, %39
  %44 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %43)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %45 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %46 = add nsw i32 %45, 1
  store i32 %46, ptr %30, align 8, !tbaa !6
  %47 = add nsw i32 %45, 2
  store i32 %47, ptr %29, align 8, !tbaa !6
  %48 = add nsw i32 %45, 3
  store i32 %48, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %48, ptr %28, align 8, !tbaa !6
  %49 = add i32 %47, %48
  %50 = add i32 %49, %46
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %50)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %52 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %53 = add nsw i32 %52, 1
  store i32 %53, ptr %30, align 8, !tbaa !6
  %54 = add nsw i32 %52, 2
  store i32 %54, ptr %29, align 8, !tbaa !6
  %55 = add nsw i32 %52, 3
  store i32 %55, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %55, ptr %28, align 8, !tbaa !6
  %56 = add i32 %54, %55
  %57 = add i32 %56, %53
  %58 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %57)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %59 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %60 = add nsw i32 %59, 1
  store i32 %60, ptr %30, align 8, !tbaa !6
  %61 = add nsw i32 %59, 2
  store i32 %61, ptr %29, align 8, !tbaa !6
  %62 = add nsw i32 %59, 3
  store i32 %62, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %62, ptr %28, align 8, !tbaa !6
  %63 = add i32 %61, %62
  %64 = add i32 %63, %60
  %65 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %64)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %66 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %67 = add nsw i32 %66, 1
  store i32 %67, ptr %30, align 8, !tbaa !6
  %68 = add nsw i32 %66, 2
  store i32 %68, ptr %29, align 8, !tbaa !6
  %69 = add nsw i32 %66, 3
  store i32 %69, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %69, ptr %28, align 8, !tbaa !6
  %70 = add i32 %68, %69
  %71 = add i32 %70, %67
  %72 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %71)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %73 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %74 = add nsw i32 %73, 1
  store i32 %74, ptr %30, align 8, !tbaa !6
  %75 = add nsw i32 %73, 2
  store i32 %75, ptr %29, align 8, !tbaa !6
  %76 = add nsw i32 %73, 3
  store i32 %76, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %76, ptr %28, align 8, !tbaa !6
  %77 = add i32 %75, %76
  %78 = add i32 %77, %74
  %79 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %78)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %80 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %81 = add nsw i32 %80, 1
  store i32 %81, ptr %30, align 8, !tbaa !6
  %82 = add nsw i32 %80, 2
  store i32 %82, ptr %29, align 8, !tbaa !6
  %83 = add nsw i32 %80, 3
  store i32 %83, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %83, ptr %28, align 8, !tbaa !6
  %84 = add i32 %82, %83
  %85 = add i32 %84, %81
  %86 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %85)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %87 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %88 = add nsw i32 %87, 1
  store i32 %88, ptr %30, align 8, !tbaa !6
  %89 = add nsw i32 %87, 2
  store i32 %89, ptr %29, align 8, !tbaa !6
  %90 = add nsw i32 %87, 3
  store i32 %90, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %90, ptr %28, align 8, !tbaa !6
  %91 = add i32 %89, %90
  %92 = add i32 %91, %88
  %93 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %92)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %94 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %95 = add nsw i32 %94, 1
  store i32 %95, ptr %30, align 8, !tbaa !6
  %96 = add nsw i32 %94, 2
  store i32 %96, ptr %29, align 8, !tbaa !6
  %97 = add nsw i32 %94, 3
  store i32 %97, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %97, ptr %28, align 8, !tbaa !6
  %98 = add i32 %96, %97
  %99 = add i32 %98, %95
  br label %210

100:                                              ; preds = %20
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %101 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %102 = add nsw i32 %101, 1
  store i32 %102, ptr %24, align 8, !tbaa !6
  %103 = add nsw i32 %101, 2
  store i32 %103, ptr %25, align 8, !tbaa !6
  %104 = add nsw i32 %101, 3
  store i32 %104, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %104, ptr %26, align 8, !tbaa !6
  %105 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %106 = load i32, ptr %24, align 8, !tbaa !6
  %107 = load i32, ptr %25, align 8, !tbaa !6
  %108 = load i32, ptr %26, align 8, !tbaa !6
  %109 = add i32 %107, %108
  %110 = add i32 %109, %106
  %111 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %110)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %112 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %113 = add nsw i32 %112, 1
  store i32 %113, ptr %24, align 8, !tbaa !6
  %114 = add nsw i32 %112, 2
  store i32 %114, ptr %25, align 8, !tbaa !6
  %115 = add nsw i32 %112, 3
  store i32 %115, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %115, ptr %26, align 8, !tbaa !6
  %116 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %117 = load i32, ptr %24, align 8, !tbaa !6
  %118 = load i32, ptr %25, align 8, !tbaa !6
  %119 = load i32, ptr %26, align 8, !tbaa !6
  %120 = add i32 %118, %119
  %121 = add i32 %120, %117
  %122 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %121)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %123 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %124 = add nsw i32 %123, 1
  store i32 %124, ptr %24, align 8, !tbaa !6
  %125 = add nsw i32 %123, 2
  store i32 %125, ptr %25, align 8, !tbaa !6
  %126 = add nsw i32 %123, 3
  store i32 %126, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %126, ptr %26, align 8, !tbaa !6
  %127 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %128 = load i32, ptr %24, align 8, !tbaa !6
  %129 = load i32, ptr %25, align 8, !tbaa !6
  %130 = load i32, ptr %26, align 8, !tbaa !6
  %131 = add i32 %129, %130
  %132 = add i32 %131, %128
  %133 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %132)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %134 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %135 = add nsw i32 %134, 1
  store i32 %135, ptr %24, align 8, !tbaa !6
  %136 = add nsw i32 %134, 2
  store i32 %136, ptr %25, align 8, !tbaa !6
  %137 = add nsw i32 %134, 3
  store i32 %137, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %137, ptr %26, align 8, !tbaa !6
  %138 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %139 = load i32, ptr %24, align 8, !tbaa !6
  %140 = load i32, ptr %25, align 8, !tbaa !6
  %141 = load i32, ptr %26, align 8, !tbaa !6
  %142 = add i32 %140, %141
  %143 = add i32 %142, %139
  %144 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %143)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %145 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %146 = add nsw i32 %145, 1
  store i32 %146, ptr %24, align 8, !tbaa !6
  %147 = add nsw i32 %145, 2
  store i32 %147, ptr %25, align 8, !tbaa !6
  %148 = add nsw i32 %145, 3
  store i32 %148, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %148, ptr %26, align 8, !tbaa !6
  %149 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %150 = load i32, ptr %24, align 8, !tbaa !6
  %151 = load i32, ptr %25, align 8, !tbaa !6
  %152 = load i32, ptr %26, align 8, !tbaa !6
  %153 = add i32 %151, %152
  %154 = add i32 %153, %150
  %155 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %154)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %156 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %157 = add nsw i32 %156, 1
  store i32 %157, ptr %24, align 8, !tbaa !6
  %158 = add nsw i32 %156, 2
  store i32 %158, ptr %25, align 8, !tbaa !6
  %159 = add nsw i32 %156, 3
  store i32 %159, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %159, ptr %26, align 8, !tbaa !6
  %160 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %161 = load i32, ptr %24, align 8, !tbaa !6
  %162 = load i32, ptr %25, align 8, !tbaa !6
  %163 = load i32, ptr %26, align 8, !tbaa !6
  %164 = add i32 %162, %163
  %165 = add i32 %164, %161
  %166 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %165)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %167 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %168 = add nsw i32 %167, 1
  store i32 %168, ptr %24, align 8, !tbaa !6
  %169 = add nsw i32 %167, 2
  store i32 %169, ptr %25, align 8, !tbaa !6
  %170 = add nsw i32 %167, 3
  store i32 %170, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %170, ptr %26, align 8, !tbaa !6
  %171 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %172 = load i32, ptr %24, align 8, !tbaa !6
  %173 = load i32, ptr %25, align 8, !tbaa !6
  %174 = load i32, ptr %26, align 8, !tbaa !6
  %175 = add i32 %173, %174
  %176 = add i32 %175, %172
  %177 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %176)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %178 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %179 = add nsw i32 %178, 1
  store i32 %179, ptr %24, align 8, !tbaa !6
  %180 = add nsw i32 %178, 2
  store i32 %180, ptr %25, align 8, !tbaa !6
  %181 = add nsw i32 %178, 3
  store i32 %181, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %181, ptr %26, align 8, !tbaa !6
  %182 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %183 = load i32, ptr %24, align 8, !tbaa !6
  %184 = load i32, ptr %25, align 8, !tbaa !6
  %185 = load i32, ptr %26, align 8, !tbaa !6
  %186 = add i32 %184, %185
  %187 = add i32 %186, %183
  %188 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %187)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %189 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %190 = add nsw i32 %189, 1
  store i32 %190, ptr %24, align 8, !tbaa !6
  %191 = add nsw i32 %189, 2
  store i32 %191, ptr %25, align 8, !tbaa !6
  %192 = add nsw i32 %189, 3
  store i32 %192, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %192, ptr %26, align 8, !tbaa !6
  %193 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %194 = load i32, ptr %24, align 8, !tbaa !6
  %195 = load i32, ptr %25, align 8, !tbaa !6
  %196 = load i32, ptr %26, align 8, !tbaa !6
  %197 = add i32 %195, %196
  %198 = add i32 %197, %194
  %199 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %198)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.start.p0(ptr nonnull %5) #3
  %200 = load i32, ptr @testAllocaOrder.count, align 4, !tbaa !13
  %201 = add nsw i32 %200, 1
  store i32 %201, ptr %24, align 8, !tbaa !6
  %202 = add nsw i32 %200, 2
  store i32 %202, ptr %25, align 8, !tbaa !6
  %203 = add nsw i32 %200, 3
  store i32 %203, ptr @testAllocaOrder.count, align 4, !tbaa !13
  store i32 %203, ptr %26, align 8, !tbaa !6
  %204 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, ptr noundef nonnull %3, ptr noundef nonnull %4, ptr noundef nonnull %5)
  %205 = load i32, ptr %24, align 8, !tbaa !6
  %206 = load i32, ptr %25, align 8, !tbaa !6
  %207 = load i32, ptr %26, align 8, !tbaa !6
  %208 = add i32 %206, %207
  %209 = add i32 %208, %205
  br label %210

210:                                              ; preds = %27, %100
  %211 = phi i32 [ %99, %27 ], [ %209, %100 ]
  %212 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %211)
  call void @llvm.lifetime.end.p0(ptr nonnull %5) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %4) #3
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #3
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !12, i64 8}
!7 = !{!"DummyStruct", !8, i64 0, !12, i64 8}
!8 = !{!"p1 _ZTS11DummyStruct", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!"int", !10, i64 0}
!13 = !{!12, !12, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"p1 omnipotent char", !9, i64 0}
