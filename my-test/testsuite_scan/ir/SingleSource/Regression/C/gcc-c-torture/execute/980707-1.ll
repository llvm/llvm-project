; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/980707-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/980707-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@buildargv.arglist = internal global [256 x ptr] zeroinitializer, align 8
@.str = private unnamed_addr constant [5 x i8] c" a b\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local noundef nonnull ptr @buildargv(ptr noundef %0) local_unnamed_addr #0 {
  br label %2

2:                                                ; preds = %18, %1
  %3 = phi i64 [ %11, %18 ], [ 0, %1 ]
  %4 = phi ptr [ %19, %18 ], [ %0, %1 ]
  br label %5

5:                                                ; preds = %8, %2
  %6 = phi ptr [ %4, %2 ], [ %9, %8 ]
  %7 = load i8, ptr %6, align 1, !tbaa !6
  switch i8 %7, label %10 [
    i8 32, label %8
    i8 0, label %20
  ]

8:                                                ; preds = %5
  %9 = getelementptr inbounds nuw i8, ptr %6, i64 1
  br label %5, !llvm.loop !9

10:                                               ; preds = %5
  %11 = add nuw nsw i64 %3, 1
  %12 = getelementptr inbounds nuw ptr, ptr @buildargv.arglist, i64 %3
  store ptr %6, ptr %12, align 8, !tbaa !11
  br label %13

13:                                               ; preds = %16, %10
  %14 = phi ptr [ %6, %10 ], [ %17, %16 ]
  %15 = load i8, ptr %14, align 1, !tbaa !6
  switch i8 %15, label %16 [
    i8 0, label %20
    i8 32, label %18
  ]

16:                                               ; preds = %13
  %17 = getelementptr inbounds nuw i8, ptr %14, i64 1
  br label %13, !llvm.loop !14

18:                                               ; preds = %13
  %19 = getelementptr inbounds nuw i8, ptr %14, i64 1
  store i8 0, ptr %14, align 1, !tbaa !6
  br label %2

20:                                               ; preds = %5, %13
  %21 = phi i64 [ %11, %13 ], [ %3, %5 ]
  %22 = shl i64 %21, 32
  %23 = ashr exact i64 %22, 29
  %24 = getelementptr inbounds i8, ptr @buildargv.arglist, i64 %23
  store ptr null, ptr %24, align 8, !tbaa !11
  ret ptr @buildargv.arglist
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [256 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #6
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(5) %1, ptr noundef nonnull align 1 dereferenceable(5) @.str, i64 5, i1 false) #6
  br label %2

2:                                                ; preds = %18, %0
  %3 = phi i64 [ %11, %18 ], [ 0, %0 ]
  %4 = phi ptr [ %19, %18 ], [ %1, %0 ]
  br label %5

5:                                                ; preds = %8, %2
  %6 = phi ptr [ %4, %2 ], [ %9, %8 ]
  %7 = load i8, ptr %6, align 1, !tbaa !6
  switch i8 %7, label %10 [
    i8 32, label %8
    i8 0, label %20
  ]

8:                                                ; preds = %5
  %9 = getelementptr inbounds nuw i8, ptr %6, i64 1
  br label %5, !llvm.loop !9

10:                                               ; preds = %5
  %11 = add nuw nsw i64 %3, 1
  %12 = getelementptr inbounds nuw ptr, ptr @buildargv.arglist, i64 %3
  store ptr %6, ptr %12, align 8, !tbaa !11
  br label %13

13:                                               ; preds = %16, %10
  %14 = phi ptr [ %6, %10 ], [ %17, %16 ]
  %15 = load i8, ptr %14, align 1, !tbaa !6
  switch i8 %15, label %16 [
    i8 0, label %20
    i8 32, label %18
  ]

16:                                               ; preds = %13
  %17 = getelementptr inbounds nuw i8, ptr %14, i64 1
  br label %13, !llvm.loop !14

18:                                               ; preds = %13
  %19 = getelementptr inbounds nuw i8, ptr %14, i64 1
  store i8 0, ptr %14, align 1, !tbaa !6
  br label %2

20:                                               ; preds = %5, %13
  %21 = phi i64 [ %11, %13 ], [ %3, %5 ]
  %22 = shl i64 %21, 32
  %23 = ashr exact i64 %22, 29
  %24 = getelementptr inbounds i8, ptr @buildargv.arglist, i64 %23
  store ptr null, ptr %24, align 8, !tbaa !11
  %25 = load ptr, ptr @buildargv.arglist, align 8, !tbaa !11
  %26 = load i8, ptr %25, align 1
  %27 = icmp eq i8 %26, 97
  br i1 %27, label %28, label %32

28:                                               ; preds = %20
  %29 = getelementptr inbounds nuw i8, ptr %25, i64 1
  %30 = load i8, ptr %29, align 1
  %31 = icmp eq i8 %30, 0
  br i1 %31, label %33, label %32

32:                                               ; preds = %20, %28
  call void @abort() #7
  unreachable

33:                                               ; preds = %28
  %34 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @buildargv.arglist, i64 8), align 8, !tbaa !11
  %35 = load i8, ptr %34, align 1
  %36 = icmp eq i8 %35, 98
  br i1 %36, label %37, label %41

37:                                               ; preds = %33
  %38 = getelementptr inbounds nuw i8, ptr %34, i64 1
  %39 = load i8, ptr %38, align 1
  %40 = icmp eq i8 %39, 0
  br i1 %40, label %42, label %41

41:                                               ; preds = %33, %37
  call void @abort() #7
  unreachable

42:                                               ; preds = %37
  %43 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @buildargv.arglist, i64 16), align 8, !tbaa !11
  %44 = icmp eq ptr %43, null
  br i1 %44, label %46, label %45

45:                                               ; preds = %42
  call void @abort() #7
  unreachable

46:                                               ; preds = %42
  call void @exit(i32 noundef 0) #8
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #4

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #5

attributes #0 = { nofree norecurse nosync nounwind memory(write, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { nounwind }
attributes #7 = { cold noreturn nounwind }
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
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = !{!12, !12, i64 0}
!12 = !{!"p1 omnipotent char", !13, i64 0}
!13 = !{!"any pointer", !7, i64 0}
!14 = distinct !{!14, !10}
