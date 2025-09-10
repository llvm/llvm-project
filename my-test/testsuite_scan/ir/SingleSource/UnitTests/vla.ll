; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/vla.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/vla.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@bork = dso_local local_unnamed_addr global [4 x [3 x i32]] [[3 x i32] [i32 1, i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6], [3 x i32] [i32 7, i32 8, i32 9], [3 x i32] [i32 10, i32 11, i32 12]], align 4
@bork2 = dso_local local_unnamed_addr global [2 x [3 x [4 x i32]]] [[3 x [4 x i32]] [[4 x i32] [i32 1, i32 2, i32 3, i32 4], [4 x i32] [i32 5, i32 6, i32 7, i32 8], [4 x i32] [i32 9, i32 10, i32 11, i32 12]], [3 x [4 x i32]] [[4 x i32] [i32 13, i32 14, i32 15, i32 16], [4 x i32] [i32 17, i32 18, i32 19, i32 20], [4 x i32] [i32 21, i32 22, i32 23, i32 24]]], align 4

; Function Attrs: nofree nounwind uwtable
define dso_local void @function(i16 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = zext i16 %0 to i64
  %4 = icmp sgt i16 %0, 0
  br i1 %4, label %5, label %56

5:                                                ; preds = %2
  %6 = zext nneg i16 %0 to i64
  br label %10

7:                                                ; preds = %10
  %8 = add nuw nsw i64 %11, 1
  %9 = icmp eq i64 %8, %6
  br i1 %9, label %17, label %10, !llvm.loop !6

10:                                               ; preds = %5, %7
  %11 = phi i64 [ 0, %5 ], [ %8, %7 ]
  %12 = getelementptr inbounds nuw i32, ptr @bork, i64 %11
  %13 = load i32, ptr %12, align 4, !tbaa !8
  %14 = getelementptr inbounds nuw i32, ptr %1, i64 %11
  %15 = load i32, ptr %14, align 4, !tbaa !8
  %16 = icmp eq i32 %13, %15
  br i1 %16, label %7, label %55

17:                                               ; preds = %7
  %18 = getelementptr inbounds nuw i32, ptr %1, i64 %3
  br label %19

19:                                               ; preds = %26, %17
  %20 = phi i64 [ 0, %17 ], [ %27, %26 ]
  %21 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @bork, i64 12), i64 %20
  %22 = load i32, ptr %21, align 4, !tbaa !8
  %23 = getelementptr inbounds nuw i32, ptr %18, i64 %20
  %24 = load i32, ptr %23, align 4, !tbaa !8
  %25 = icmp eq i32 %22, %24
  br i1 %25, label %26, label %55

26:                                               ; preds = %19
  %27 = add nuw nsw i64 %20, 1
  %28 = icmp eq i64 %27, %6
  br i1 %28, label %29, label %19, !llvm.loop !6

29:                                               ; preds = %26
  %30 = shl nuw nsw i64 %3, 3
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 %30
  br label %32

32:                                               ; preds = %39, %29
  %33 = phi i64 [ 0, %29 ], [ %40, %39 ]
  %34 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @bork, i64 24), i64 %33
  %35 = load i32, ptr %34, align 4, !tbaa !8
  %36 = getelementptr inbounds nuw i32, ptr %31, i64 %33
  %37 = load i32, ptr %36, align 4, !tbaa !8
  %38 = icmp eq i32 %35, %37
  br i1 %38, label %39, label %55

39:                                               ; preds = %32
  %40 = add nuw nsw i64 %33, 1
  %41 = icmp eq i64 %40, %6
  br i1 %41, label %42, label %32, !llvm.loop !6

42:                                               ; preds = %39
  %43 = mul nuw nsw i64 %3, 12
  %44 = getelementptr inbounds nuw i8, ptr %1, i64 %43
  br label %45

45:                                               ; preds = %52, %42
  %46 = phi i64 [ 0, %42 ], [ %53, %52 ]
  %47 = getelementptr inbounds nuw i32, ptr getelementptr inbounds nuw (i8, ptr @bork, i64 36), i64 %46
  %48 = load i32, ptr %47, align 4, !tbaa !8
  %49 = getelementptr inbounds nuw i32, ptr %44, i64 %46
  %50 = load i32, ptr %49, align 4, !tbaa !8
  %51 = icmp eq i32 %48, %50
  br i1 %51, label %52, label %55

52:                                               ; preds = %45
  %53 = add nuw nsw i64 %46, 1
  %54 = icmp eq i64 %53, %6
  br i1 %54, label %56, label %45, !llvm.loop !6

55:                                               ; preds = %10, %19, %32, %45
  tail call void @abort() #3
  unreachable

56:                                               ; preds = %52, %2
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @test() local_unnamed_addr #2 {
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local void @function2(i16 noundef %0, i16 noundef %1, ptr noundef readonly captures(none) %2) local_unnamed_addr #0 {
  %4 = zext i16 %0 to i64
  %5 = zext i16 %1 to i64
  %6 = sext i16 %0 to i64
  %7 = icmp sgt i16 %0, 0
  %8 = sext i16 %1 to i64
  %9 = mul nuw nsw i64 %5, %4
  %10 = icmp sgt i16 %1, 0
  %11 = and i1 %7, %10
  br i1 %11, label %12, label %51

12:                                               ; preds = %3, %27
  %13 = phi i64 [ %28, %27 ], [ 0, %3 ]
  %14 = getelementptr inbounds nuw [4 x i32], ptr @bork2, i64 %13
  %15 = mul nuw nsw i64 %13, %5
  %16 = getelementptr inbounds nuw i32, ptr %2, i64 %15
  br label %20

17:                                               ; preds = %20
  %18 = add nuw nsw i64 %21, 1
  %19 = icmp eq i64 %18, %8
  br i1 %19, label %27, label %20, !llvm.loop !12

20:                                               ; preds = %17, %12
  %21 = phi i64 [ %18, %17 ], [ 0, %12 ]
  %22 = getelementptr inbounds nuw i32, ptr %14, i64 %21
  %23 = load i32, ptr %22, align 4, !tbaa !8
  %24 = getelementptr inbounds nuw i32, ptr %16, i64 %21
  %25 = load i32, ptr %24, align 4, !tbaa !8
  %26 = icmp eq i32 %23, %25
  br i1 %26, label %17, label %50

27:                                               ; preds = %17
  %28 = add nuw nsw i64 %13, 1
  %29 = icmp eq i64 %28, %6
  br i1 %29, label %30, label %12, !llvm.loop !13

30:                                               ; preds = %27
  %31 = getelementptr inbounds nuw i32, ptr %2, i64 %9
  br label %32

32:                                               ; preds = %47, %30
  %33 = phi i64 [ %48, %47 ], [ 0, %30 ]
  %34 = getelementptr inbounds nuw [4 x i32], ptr getelementptr inbounds nuw (i8, ptr @bork2, i64 48), i64 %33
  %35 = mul nuw nsw i64 %33, %5
  %36 = getelementptr inbounds nuw i32, ptr %31, i64 %35
  br label %37

37:                                               ; preds = %44, %32
  %38 = phi i64 [ %45, %44 ], [ 0, %32 ]
  %39 = getelementptr inbounds nuw i32, ptr %34, i64 %38
  %40 = load i32, ptr %39, align 4, !tbaa !8
  %41 = getelementptr inbounds nuw i32, ptr %36, i64 %38
  %42 = load i32, ptr %41, align 4, !tbaa !8
  %43 = icmp eq i32 %40, %42
  br i1 %43, label %44, label %50

44:                                               ; preds = %37
  %45 = add nuw nsw i64 %38, 1
  %46 = icmp eq i64 %45, %8
  br i1 %46, label %47, label %37, !llvm.loop !12

47:                                               ; preds = %44
  %48 = add nuw nsw i64 %33, 1
  %49 = icmp eq i64 %48, %6
  br i1 %49, label %51, label %32, !llvm.loop !13

50:                                               ; preds = %20, %37
  tail call void @abort() #3
  unreachable

51:                                               ; preds = %47, %3
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local void @test2() local_unnamed_addr #2 {
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  ret i32 0
}

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
