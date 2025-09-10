; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr45070.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr45070.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.source = type { i32, i32, i32 }

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca %struct.source, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  store <2 x i32> zeroinitializer, ptr %1, align 8, !tbaa !6
  %2 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store i32 0, ptr %2, align 8, !tbaa !10
  %3 = call fastcc i16 @next(ptr noundef %1)
  %4 = icmp eq i16 %3, -1
  br i1 %4, label %5, label %51

5:                                                ; preds = %0
  %6 = call fastcc i16 @next(ptr noundef %1)
  %7 = icmp eq i16 %6, 0
  br i1 %7, label %8, label %51

8:                                                ; preds = %5
  %9 = call fastcc i16 @next(ptr noundef %1)
  %10 = icmp eq i16 %9, 0
  br i1 %10, label %11, label %51

11:                                               ; preds = %8
  %12 = call fastcc i16 @next(ptr noundef %1)
  %13 = icmp eq i16 %12, 0
  br i1 %13, label %14, label %51

14:                                               ; preds = %11
  %15 = call fastcc i16 @next(ptr noundef %1)
  %16 = icmp eq i16 %15, 0
  br i1 %16, label %17, label %51

17:                                               ; preds = %14
  %18 = call fastcc i16 @next(ptr noundef %1)
  %19 = icmp eq i16 %18, 0
  br i1 %19, label %20, label %51

20:                                               ; preds = %17
  %21 = call fastcc i16 @next(ptr noundef %1)
  %22 = icmp eq i16 %21, 0
  br i1 %22, label %23, label %51

23:                                               ; preds = %20
  %24 = call fastcc i16 @next(ptr noundef %1)
  %25 = icmp eq i16 %24, 0
  br i1 %25, label %26, label %51

26:                                               ; preds = %23
  %27 = call fastcc i16 @next(ptr noundef %1)
  %28 = icmp eq i16 %27, 0
  br i1 %28, label %29, label %51

29:                                               ; preds = %26
  %30 = call fastcc i16 @next(ptr noundef %1)
  %31 = icmp eq i16 %30, 0
  br i1 %31, label %32, label %51

32:                                               ; preds = %29
  %33 = call fastcc i16 @next(ptr noundef %1)
  %34 = icmp eq i16 %33, 0
  br i1 %34, label %35, label %51

35:                                               ; preds = %32
  %36 = call fastcc i16 @next(ptr noundef %1)
  %37 = icmp eq i16 %36, 0
  br i1 %37, label %38, label %51

38:                                               ; preds = %35
  %39 = call fastcc i16 @next(ptr noundef %1)
  %40 = icmp eq i16 %39, 0
  br i1 %40, label %41, label %51

41:                                               ; preds = %38
  %42 = call fastcc i16 @next(ptr noundef %1)
  %43 = icmp eq i16 %42, 0
  br i1 %43, label %44, label %51

44:                                               ; preds = %41
  %45 = call fastcc i16 @next(ptr noundef %1)
  %46 = icmp eq i16 %45, 0
  br i1 %46, label %47, label %51

47:                                               ; preds = %44
  %48 = call fastcc i16 @next(ptr noundef %1)
  %49 = icmp eq i16 %48, 0
  br i1 %49, label %50, label %51

50:                                               ; preds = %47
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0

51:                                               ; preds = %47, %44, %41, %38, %35, %32, %29, %26, %23, %20, %17, %14, %11, %8, %5, %0
  tail call void @abort() #6
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable
define internal fastcc range(i16 -1, 1) i16 @next(ptr noundef nonnull captures(none) %0) unnamed_addr #2 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %3 = load i32, ptr %0, align 4, !tbaa !12
  %4 = load i32, ptr %2, align 4, !tbaa !13
  %5 = icmp slt i32 %3, %4
  br i1 %5, label %16, label %6

6:                                                ; preds = %1
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %8

8:                                                ; preds = %6, %11
  %9 = load i32, ptr %7, align 4, !tbaa !10
  %10 = icmp eq i32 %9, 0
  br i1 %10, label %15, label %11

11:                                               ; preds = %8
  store i32 0, ptr %7, align 4, !tbaa !10
  tail call fastcc void @fetch(ptr noundef %0)
  %12 = load i32, ptr %0, align 4, !tbaa !12
  %13 = load i32, ptr %2, align 4, !tbaa !13
  %14 = icmp slt i32 %12, %13
  br i1 %14, label %16, label %8

15:                                               ; preds = %8
  store i32 1, ptr %7, align 4, !tbaa !10
  br label %16

16:                                               ; preds = %11, %1, %15
  %17 = phi i16 [ -1, %15 ], [ 0, %1 ], [ 0, %11 ]
  ret i16 %17
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define internal fastcc void @fetch(ptr noundef nonnull writeonly captures(none) initializes((4, 8)) %0) unnamed_addr #4 {
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 128, ptr %2, align 4, !tbaa !13
  ret void
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree noinline norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

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
!10 = !{!11, !7, i64 8}
!11 = !{!"source", !7, i64 0, !7, i64 4, !7, i64 8}
!12 = !{!11, !7, i64 0}
!13 = !{!11, !7, i64 4}
