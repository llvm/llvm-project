; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr20601-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr20601-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.T = type { ptr, [4096 x i8], ptr }

@.str = private unnamed_addr constant [2 x i8] c"a\00", align 1
@.str.1 = private unnamed_addr constant [3 x i8] c"-u\00", align 1
@.str.2 = private unnamed_addr constant [2 x i8] c"b\00", align 1
@.str.3 = private unnamed_addr constant [2 x i8] c"c\00", align 1
@g = dso_local global [4 x ptr] [ptr @.str, ptr @.str.1, ptr @.str.2, ptr @.str.3], align 8
@c = dso_local local_unnamed_addr global ptr null, align 8
@b = dso_local local_unnamed_addr global i32 0, align 4
@.str.4 = private unnamed_addr constant [8 x i8] c"/bin/sh\00", align 1
@t = dso_local local_unnamed_addr global %struct.T zeroinitializer, align 8
@a = dso_local local_unnamed_addr global [5 x i32] zeroinitializer, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global ptr null, align 8
@f = dso_local global [16 x ptr] zeroinitializer, align 8

; Function Attrs: nofree norecurse noreturn nosync nounwind memory(none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  br label %1

1:                                                ; preds = %0, %1
  br label %1
}

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noalias noundef ptr @bar(ptr noundef readnone captures(none) %0, i32 noundef %1) local_unnamed_addr #1 {
  ret ptr null
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  store ptr @g, ptr @c, align 8, !tbaa !6
  store i32 4, ptr @b, align 4, !tbaa !12
  store ptr getelementptr inbounds nuw (i8, ptr @g, i64 8), ptr @e, align 8, !tbaa !6
  store i32 3, ptr @d, align 4, !tbaa !12
  br label %1

1:                                                ; preds = %0, %32
  %2 = phi i32 [ %35, %32 ], [ 1, %0 ]
  %3 = phi ptr [ %37, %32 ], [ getelementptr inbounds nuw (i8, ptr @g, i64 8), %0 ]
  %4 = phi i32 [ %36, %32 ], [ 3, %0 ]
  %5 = load ptr, ptr %3, align 8, !tbaa !14
  %6 = load i8, ptr %5, align 1, !tbaa !16
  %7 = icmp eq i8 %6, 45
  br i1 %7, label %8, label %39

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw i8, ptr %5, i64 1
  %10 = load i8, ptr %9, align 1, !tbaa !16
  %11 = icmp eq i8 %10, 0
  br i1 %11, label %32, label %12

12:                                               ; preds = %8
  %13 = getelementptr inbounds nuw i8, ptr %5, i64 2
  %14 = load i8, ptr %13, align 1, !tbaa !16
  %15 = icmp eq i8 %14, 0
  br i1 %15, label %17, label %16

16:                                               ; preds = %12
  tail call void @abort() #5
  unreachable

17:                                               ; preds = %12
  switch i8 %10, label %32 [
    i8 117, label %18
    i8 80, label %25
    i8 45, label %27
  ]

18:                                               ; preds = %17
  %19 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %20 = load ptr, ptr %19, align 8, !tbaa !14
  %21 = icmp eq ptr %20, null
  br i1 %21, label %22, label %23

22:                                               ; preds = %18
  tail call void @abort() #5
  unreachable

23:                                               ; preds = %18
  store ptr %19, ptr getelementptr inbounds nuw (i8, ptr @t, i64 4104), align 8, !tbaa !17
  %24 = add nsw i32 %4, -1
  br label %32

25:                                               ; preds = %17
  %26 = or i32 %2, 4096
  br label %32

27:                                               ; preds = %17
  %28 = add nsw i32 %4, -1
  store i32 %28, ptr @d, align 4, !tbaa !12
  %29 = getelementptr inbounds nuw i8, ptr %3, i64 8
  store ptr %29, ptr @e, align 8, !tbaa !6
  %30 = icmp eq i32 %2, 1
  %31 = select i1 %30, i32 1537, i32 %2
  br label %43

32:                                               ; preds = %25, %23, %17, %8
  %33 = phi i32 [ %4, %17 ], [ %24, %23 ], [ %4, %25 ], [ %4, %8 ]
  %34 = phi ptr [ %3, %17 ], [ %19, %23 ], [ %3, %25 ], [ %3, %8 ]
  %35 = phi i32 [ %2, %17 ], [ %2, %23 ], [ %26, %25 ], [ %2, %8 ]
  %36 = add nsw i32 %33, -1
  store i32 %36, ptr @d, align 4, !tbaa !12
  %37 = getelementptr inbounds nuw i8, ptr %34, i64 8
  store ptr %37, ptr @e, align 8, !tbaa !6
  %38 = icmp sgt i32 %33, 1
  br i1 %38, label %1, label %43, !llvm.loop !19

39:                                               ; preds = %1
  %40 = and i32 %2, 1
  %41 = icmp eq i32 %40, 0
  br i1 %41, label %42, label %43

42:                                               ; preds = %39
  tail call void @abort() #5
  unreachable

43:                                               ; preds = %32, %27, %39
  %44 = phi i32 [ %28, %27 ], [ %4, %39 ], [ %36, %32 ]
  %45 = phi ptr [ %29, %27 ], [ %3, %39 ], [ %37, %32 ]
  %46 = phi i32 [ %31, %27 ], [ %2, %39 ], [ %35, %32 ]
  store ptr @.str.4, ptr @t, align 8, !tbaa !21
  %47 = and i32 %46, 512
  %48 = icmp eq i32 %47, 0
  br i1 %48, label %58, label %49

49:                                               ; preds = %43
  %50 = add nsw i32 %44, 1
  store i32 %50, ptr @d, align 4, !tbaa !12
  store ptr @f, ptr @e, align 8, !tbaa !6
  store ptr @.str.4, ptr @f, align 8, !tbaa !14
  br label %51

51:                                               ; preds = %51, %49
  %52 = phi ptr [ @f, %49 ], [ %54, %51 ]
  %53 = phi ptr [ %45, %49 ], [ %57, %51 ]
  %54 = getelementptr inbounds nuw i8, ptr %52, i64 8
  %55 = load ptr, ptr %53, align 8, !tbaa !14
  store ptr %55, ptr %54, align 8, !tbaa !14
  %56 = icmp eq ptr %55, null
  %57 = getelementptr inbounds nuw i8, ptr %53, i64 8
  br i1 %56, label %58, label %51, !llvm.loop !22

58:                                               ; preds = %51, %43
  %59 = and i32 %46, 1024
  %60 = icmp eq i32 %59, 0
  %61 = load i32, ptr getelementptr inbounds nuw (i8, ptr @a, i64 16), align 4
  %62 = icmp ne i32 %61, 0
  %63 = select i1 %60, i1 true, i1 %62
  br i1 %63, label %65, label %64

64:                                               ; preds = %58
  tail call void @abort() #5
  unreachable

65:                                               ; preds = %58
  tail call void @exit(i32 noundef 0) #5
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #4

attributes #0 = { nofree norecurse noreturn nosync nounwind memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p2 omnipotent char", !8, i64 0}
!8 = !{!"any p2 pointer", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !10, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"p1 omnipotent char", !9, i64 0}
!16 = !{!10, !10, i64 0}
!17 = !{!18, !7, i64 4104}
!18 = !{!"T", !15, i64 0, !10, i64 8, !7, i64 4104}
!19 = distinct !{!19, !20}
!20 = !{!"llvm.loop.mustprogress"}
!21 = !{!18, !15, i64 0}
!22 = distinct !{!22, !20}
