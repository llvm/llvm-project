; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr49419.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr49419.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { i32, i32, i32 }

@t = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local range(i32 -2147483647, -2147483648) i32 @foo(i32 noundef %0, i32 noundef %1, ptr noundef writeonly captures(none) %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp eq i32 %0, -1
  br i1 %5, label %48, label %6

6:                                                ; preds = %4
  %7 = load ptr, ptr @t, align 8, !tbaa !6
  %8 = sext i32 %0 to i64
  %9 = getelementptr inbounds %struct.S, ptr %7, i64 %8
  %10 = load i32, ptr %9, align 4, !tbaa !11
  %11 = icmp eq i32 %10, %1
  %12 = icmp sgt i32 %3, 0
  %13 = and i1 %11, %12
  br i1 %13, label %14, label %28

14:                                               ; preds = %6, %14
  %15 = phi i64 [ %20, %14 ], [ %8, %6 ]
  %16 = phi i32 [ %19, %14 ], [ 0, %6 ]
  %17 = getelementptr inbounds %struct.S, ptr %7, i64 %15, i32 1
  %18 = load i32, ptr %17, align 4, !tbaa !14
  %19 = add nuw nsw i32 %16, 1
  %20 = sext i32 %18 to i64
  %21 = getelementptr inbounds %struct.S, ptr %7, i64 %20
  %22 = load i32, ptr %21, align 4, !tbaa !11
  %23 = icmp eq i32 %22, %1
  %24 = icmp slt i32 %19, %3
  %25 = select i1 %23, i1 %24, i1 false
  br i1 %25, label %14, label %26, !llvm.loop !15

26:                                               ; preds = %14
  %27 = icmp eq i32 %19, %3
  br i1 %27, label %30, label %31

28:                                               ; preds = %6
  %29 = icmp eq i32 %3, 0
  br i1 %29, label %30, label %45

30:                                               ; preds = %28, %26
  tail call void @abort() #4
  unreachable

31:                                               ; preds = %26
  %32 = add nuw nsw i32 %16, 2
  %33 = zext nneg i32 %19 to i64
  br label %34

34:                                               ; preds = %31, %34
  %35 = phi i64 [ %33, %31 ], [ %43, %34 ]
  %36 = phi i32 [ %0, %31 ], [ %42, %34 ]
  %37 = sext i32 %36 to i64
  %38 = getelementptr inbounds %struct.S, ptr %7, i64 %37, i32 2
  %39 = load i32, ptr %38, align 4, !tbaa !17
  %40 = getelementptr inbounds nuw i32, ptr %2, i64 %35
  store i32 %39, ptr %40, align 4, !tbaa !18
  %41 = getelementptr inbounds %struct.S, ptr %7, i64 %37, i32 1
  %42 = load i32, ptr %41, align 4, !tbaa !14
  %43 = add nsw i64 %35, -1
  %44 = icmp samesign ugt i64 %35, 1
  br i1 %44, label %34, label %45, !llvm.loop !19

45:                                               ; preds = %34, %28
  %46 = phi i32 [ 1, %28 ], [ %32, %34 ]
  %47 = phi i32 [ %0, %28 ], [ %42, %34 ]
  store i32 %47, ptr %2, align 4, !tbaa !18
  br label %48

48:                                               ; preds = %4, %45
  %49 = phi i32 [ %46, %45 ], [ 0, %4 ]
  ret i32 %49
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [3 x i32], align 4
  %2 = alloca [3 x %struct.S], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 12
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(24) %3, i8 0, i64 24, i1 false)
  store <2 x i32> splat (i32 1), ptr %2, align 8
  %4 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i32 2, ptr %4, align 8
  store ptr %2, ptr @t, align 8, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %6 = load i32, ptr %5, align 4, !tbaa !14
  %7 = sext i32 %6 to i64
  %8 = getelementptr inbounds %struct.S, ptr %2, i64 %7
  %9 = load i32, ptr %8, align 4, !tbaa !11
  %10 = icmp eq i32 %9, 1
  br i1 %10, label %11, label %20, !llvm.loop !15

11:                                               ; preds = %0
  %12 = getelementptr inbounds %struct.S, ptr %2, i64 %7, i32 1
  %13 = load i32, ptr %12, align 4, !tbaa !14
  %14 = sext i32 %13 to i64
  %15 = getelementptr inbounds %struct.S, ptr %2, i64 %14
  %16 = load i32, ptr %15, align 4, !tbaa !11
  %17 = icmp eq i32 %16, 1
  %18 = select i1 %17, i1 true, i1 false
  %19 = select i1 %17, i64 3, i64 2
  br label %20, !llvm.loop !15

20:                                               ; preds = %11, %0
  %21 = phi i1 [ false, %0 ], [ %18, %11 ]
  %22 = phi i64 [ 1, %0 ], [ %19, %11 ]
  br i1 %21, label %23, label %24

23:                                               ; preds = %20
  call void @abort() #4
  unreachable

24:                                               ; preds = %20
  %25 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %26 = load i32, ptr %25, align 8, !tbaa !17
  %27 = getelementptr inbounds nuw i32, ptr %1, i64 %22
  store i32 %26, ptr %27, align 4, !tbaa !18
  %28 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %29 = load i32, ptr %28, align 4, !tbaa !14
  br i1 %10, label %30, label %47, !llvm.loop !19

30:                                               ; preds = %24
  %31 = add nsw i64 %22, -1
  %32 = sext i32 %29 to i64
  %33 = getelementptr inbounds %struct.S, ptr %2, i64 %32, i32 2
  %34 = load i32, ptr %33, align 4, !tbaa !17
  %35 = getelementptr inbounds nuw i32, ptr %1, i64 %31
  store i32 %34, ptr %35, align 4, !tbaa !18
  %36 = getelementptr inbounds %struct.S, ptr %2, i64 %32, i32 1
  %37 = load i32, ptr %36, align 4, !tbaa !14
  %38 = icmp samesign ugt i64 %31, 1
  br i1 %38, label %39, label %47, !llvm.loop !19

39:                                               ; preds = %30
  %40 = sext i32 %37 to i64
  %41 = getelementptr inbounds %struct.S, ptr %2, i64 %40, i32 2
  %42 = load i32, ptr %41, align 4, !tbaa !17
  %43 = getelementptr i32, ptr %1, i64 %22
  %44 = getelementptr i8, ptr %43, i64 -8
  store i32 %42, ptr %44, align 4, !tbaa !18
  %45 = getelementptr inbounds %struct.S, ptr %2, i64 %40, i32 1
  %46 = load i32, ptr %45, align 4, !tbaa !14
  br label %47

47:                                               ; preds = %39, %30, %24
  %48 = phi i32 [ %29, %24 ], [ %37, %30 ], [ %46, %39 ]
  br i1 %10, label %49, label %50

49:                                               ; preds = %47
  call void @abort() #4
  unreachable

50:                                               ; preds = %47
  %51 = icmp ne i32 %48, 1
  %52 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %53 = load i32, ptr %52, align 4
  %54 = icmp ne i32 %53, 2
  %55 = select i1 %51, i1 true, i1 %54
  br i1 %55, label %56, label %57

56:                                               ; preds = %50
  call void @abort() #4
  unreachable

57:                                               ; preds = %50
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { noreturn nounwind }
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
!7 = !{!"p1 _ZTS1S", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !13, i64 0}
!12 = !{!"S", !13, i64 0, !13, i64 4, !13, i64 8}
!13 = !{!"int", !9, i64 0}
!14 = !{!12, !13, i64 4}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
!17 = !{!12, !13, i64 8}
!18 = !{!13, !13, i64 0}
!19 = distinct !{!19, !16}
