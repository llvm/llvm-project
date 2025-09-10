; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/PR640.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/PR640.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@str = private unnamed_addr constant [10 x i8] c"All done.\00", align 4
@str.2 = private unnamed_addr constant [6 x i8] c"ERROR\00", align 4

; Function Attrs: nofree nounwind uwtable
define dso_local range(i32 0, 2) i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %3)
  store i32 1, ptr %3, align 4, !tbaa !6
  %4 = call i32 (ptr, ...) @test_stdarg_va(ptr noundef %3, i32 noundef 1, i64 noundef 1981891429, i32 noundef 2, ptr noundef nonnull %3)
  %5 = icmp eq i32 %4, 0
  br i1 %5, label %9, label %6

6:                                                ; preds = %2
  %7 = call i32 (ptr, ...) @test_stdarg_builtin_va(ptr noundef %3, i32 noundef 1, i64 noundef 1981891433, i32 noundef 2, ptr noundef nonnull %3)
  %8 = icmp eq i32 %7, 0
  br i1 %8, label %9, label %10

9:                                                ; preds = %2, %6
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  br label %16

10:                                               ; preds = %6
  %11 = load i32, ptr %3, align 4, !tbaa !6
  %12 = and i32 %11, 1
  call void @llvm.lifetime.end.p0(ptr nonnull %3)
  %13 = icmp eq i32 %12, 0
  %14 = select i1 %13, ptr @str.2, ptr @str
  %15 = xor i32 %12, 1
  br label %16

16:                                               ; preds = %10, %9
  %17 = phi ptr [ @str.2, %9 ], [ %14, %10 ]
  %18 = phi i32 [ 1, %9 ], [ %15, %10 ]
  %19 = call i32 @puts(ptr nonnull dereferenceable(1) %17)
  ret i32 %18
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define internal range(i32 0, 2) i32 @test_stdarg_va(ptr noundef nonnull readnone captures(address) %0, ...) unnamed_addr #2 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %4 = load i32, ptr %3, align 8
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %9, label %6

6:                                                ; preds = %1
  %7 = add nsw i32 %4, 8
  store i32 %7, ptr %3, align 8
  %8 = icmp samesign ult i32 %4, -7
  br i1 %8, label %13, label %9

9:                                                ; preds = %1, %6
  %10 = load ptr, ptr %2, align 8
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 8
  store ptr %11, ptr %2, align 8
  %12 = load i32, ptr %10, align 8, !tbaa !6
  br label %23

13:                                               ; preds = %6
  %14 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %15 = load ptr, ptr %14, align 8
  %16 = sext i32 %4 to i64
  %17 = getelementptr inbounds i8, ptr %15, i64 %16
  %18 = load i32, ptr %17, align 8, !tbaa !6
  %19 = icmp sgt i32 %4, -9
  br i1 %19, label %23, label %20

20:                                               ; preds = %13
  %21 = add nsw i32 %4, 16
  store i32 %21, ptr %3, align 8
  %22 = icmp samesign ult i32 %7, -7
  br i1 %22, label %28, label %23

23:                                               ; preds = %13, %20, %9
  %24 = phi i32 [ %18, %20 ], [ %18, %13 ], [ %12, %9 ]
  %25 = load ptr, ptr %2, align 8
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 8
  store ptr %26, ptr %2, align 8
  %27 = load i64, ptr %25, align 8, !tbaa !10
  br label %38

28:                                               ; preds = %20
  %29 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %30 = load ptr, ptr %29, align 8
  %31 = sext i32 %7 to i64
  %32 = getelementptr inbounds i8, ptr %30, i64 %31
  %33 = load i64, ptr %32, align 8, !tbaa !10
  %34 = icmp sgt i32 %4, -17
  br i1 %34, label %38, label %35

35:                                               ; preds = %28
  %36 = add nsw i32 %4, 24
  store i32 %36, ptr %3, align 8
  %37 = icmp samesign ult i32 %21, -7
  br i1 %37, label %44, label %38

38:                                               ; preds = %28, %35, %23
  %39 = phi i64 [ %33, %35 ], [ %33, %28 ], [ %27, %23 ]
  %40 = phi i32 [ %18, %35 ], [ %18, %28 ], [ %24, %23 ]
  %41 = load ptr, ptr %2, align 8
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 8
  store ptr %42, ptr %2, align 8
  %43 = load i32, ptr %41, align 8, !tbaa !6
  br label %59

44:                                               ; preds = %35
  %45 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %46 = load ptr, ptr %45, align 8
  %47 = sext i32 %21 to i64
  %48 = getelementptr inbounds i8, ptr %46, i64 %47
  %49 = load i32, ptr %48, align 8, !tbaa !6
  %50 = icmp sgt i32 %4, -25
  br i1 %50, label %59, label %51

51:                                               ; preds = %44
  %52 = add nsw i32 %4, 32
  store i32 %52, ptr %3, align 8
  %53 = icmp samesign ult i32 %36, -7
  br i1 %53, label %54, label %59

54:                                               ; preds = %51
  %55 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %56 = load ptr, ptr %55, align 8
  %57 = sext i32 %36 to i64
  %58 = getelementptr inbounds i8, ptr %56, i64 %57
  br label %65

59:                                               ; preds = %38, %51, %44
  %60 = phi i32 [ %43, %38 ], [ %49, %51 ], [ %49, %44 ]
  %61 = phi i32 [ %40, %38 ], [ %18, %51 ], [ %18, %44 ]
  %62 = phi i64 [ %39, %38 ], [ %33, %51 ], [ %33, %44 ]
  %63 = load ptr, ptr %2, align 8
  %64 = getelementptr inbounds nuw i8, ptr %63, i64 8
  store ptr %64, ptr %2, align 8
  br label %65

65:                                               ; preds = %59, %54
  %66 = phi i32 [ %49, %54 ], [ %60, %59 ]
  %67 = phi i32 [ %18, %54 ], [ %61, %59 ]
  %68 = phi i64 [ %33, %54 ], [ %62, %59 ]
  %69 = phi ptr [ %58, %54 ], [ %63, %59 ]
  %70 = load ptr, ptr %69, align 8, !tbaa !12
  call void @llvm.va_end.p0(ptr nonnull %2)
  %71 = icmp eq ptr %0, %70
  %72 = icmp eq i32 %67, 1
  %73 = select i1 %71, i1 %72, i1 false
  %74 = icmp eq i64 %68, 1981891429
  %75 = select i1 %73, i1 %74, i1 false
  %76 = icmp eq i32 %66, 2
  %77 = select i1 %75, i1 %76, i1 false
  %78 = zext i1 %77 to i32
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  ret i32 %78
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn uwtable
define internal range(i32 0, 2) i32 @test_stdarg_builtin_va(ptr noundef nonnull readnone captures(address) %0, ...) unnamed_addr #2 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %4 = load i32, ptr %3, align 8
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %9, label %6

6:                                                ; preds = %1
  %7 = add nsw i32 %4, 8
  store i32 %7, ptr %3, align 8
  %8 = icmp samesign ult i32 %4, -7
  br i1 %8, label %13, label %9

9:                                                ; preds = %1, %6
  %10 = load ptr, ptr %2, align 8
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 8
  store ptr %11, ptr %2, align 8
  %12 = load i32, ptr %10, align 8, !tbaa !6
  br label %23

13:                                               ; preds = %6
  %14 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %15 = load ptr, ptr %14, align 8
  %16 = sext i32 %4 to i64
  %17 = getelementptr inbounds i8, ptr %15, i64 %16
  %18 = load i32, ptr %17, align 8, !tbaa !6
  %19 = icmp sgt i32 %4, -9
  br i1 %19, label %23, label %20

20:                                               ; preds = %13
  %21 = add nsw i32 %4, 16
  store i32 %21, ptr %3, align 8
  %22 = icmp samesign ult i32 %7, -7
  br i1 %22, label %28, label %23

23:                                               ; preds = %13, %20, %9
  %24 = phi i32 [ %18, %20 ], [ %18, %13 ], [ %12, %9 ]
  %25 = load ptr, ptr %2, align 8
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 8
  store ptr %26, ptr %2, align 8
  %27 = load i64, ptr %25, align 8, !tbaa !10
  br label %38

28:                                               ; preds = %20
  %29 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %30 = load ptr, ptr %29, align 8
  %31 = sext i32 %7 to i64
  %32 = getelementptr inbounds i8, ptr %30, i64 %31
  %33 = load i64, ptr %32, align 8, !tbaa !10
  %34 = icmp sgt i32 %4, -17
  br i1 %34, label %38, label %35

35:                                               ; preds = %28
  %36 = add nsw i32 %4, 24
  store i32 %36, ptr %3, align 8
  %37 = icmp samesign ult i32 %21, -7
  br i1 %37, label %44, label %38

38:                                               ; preds = %28, %35, %23
  %39 = phi i64 [ %33, %35 ], [ %33, %28 ], [ %27, %23 ]
  %40 = phi i32 [ %18, %35 ], [ %18, %28 ], [ %24, %23 ]
  %41 = load ptr, ptr %2, align 8
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 8
  store ptr %42, ptr %2, align 8
  %43 = load i32, ptr %41, align 8, !tbaa !6
  br label %59

44:                                               ; preds = %35
  %45 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %46 = load ptr, ptr %45, align 8
  %47 = sext i32 %21 to i64
  %48 = getelementptr inbounds i8, ptr %46, i64 %47
  %49 = load i32, ptr %48, align 8, !tbaa !6
  %50 = icmp sgt i32 %4, -25
  br i1 %50, label %59, label %51

51:                                               ; preds = %44
  %52 = add nsw i32 %4, 32
  store i32 %52, ptr %3, align 8
  %53 = icmp samesign ult i32 %36, -7
  br i1 %53, label %54, label %59

54:                                               ; preds = %51
  %55 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %56 = load ptr, ptr %55, align 8
  %57 = sext i32 %36 to i64
  %58 = getelementptr inbounds i8, ptr %56, i64 %57
  br label %65

59:                                               ; preds = %38, %51, %44
  %60 = phi i32 [ %43, %38 ], [ %49, %51 ], [ %49, %44 ]
  %61 = phi i32 [ %40, %38 ], [ %18, %51 ], [ %18, %44 ]
  %62 = phi i64 [ %39, %38 ], [ %33, %51 ], [ %33, %44 ]
  %63 = load ptr, ptr %2, align 8
  %64 = getelementptr inbounds nuw i8, ptr %63, i64 8
  store ptr %64, ptr %2, align 8
  br label %65

65:                                               ; preds = %59, %54
  %66 = phi i32 [ %49, %54 ], [ %60, %59 ]
  %67 = phi i32 [ %18, %54 ], [ %61, %59 ]
  %68 = phi i64 [ %33, %54 ], [ %62, %59 ]
  %69 = phi ptr [ %58, %54 ], [ %63, %59 ]
  %70 = load ptr, ptr %69, align 8, !tbaa !12
  call void @llvm.va_end.p0(ptr nonnull %2)
  %71 = icmp eq ptr %0, %70
  %72 = icmp eq i32 %67, 1
  %73 = select i1 %71, i1 %72, i1 false
  %74 = icmp eq i64 %68, 1981891433
  %75 = select i1 %73, i1 %74, i1 false
  %76 = icmp eq i32 %66, 2
  %77 = select i1 %75, i1 %76, i1 false
  %78 = zext i1 %77 to i32
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  ret i32 %78
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #3

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #4

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #4 = { nofree nounwind }
attributes #5 = { nounwind }

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
!12 = !{!13, !13, i64 0}
!13 = !{!"any pointer", !8, i64 0}
