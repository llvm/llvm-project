; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20010224-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20010224-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@masktab = dso_local local_unnamed_addr global [6 x i16] [i16 1, i16 2, i16 3, i16 4, i16 5, i16 0], align 4
@psd = dso_local local_unnamed_addr global [6 x i16] [i16 50, i16 40, i16 30, i16 20, i16 10, i16 0], align 8
@bndpsd = dso_local local_unnamed_addr global [6 x i16] [i16 1, i16 2, i16 3, i16 4, i16 5, i16 0], align 2

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @ba_compute_psd(i16 noundef %0) local_unnamed_addr #0 {
  %2 = sext i16 %0 to i64
  %3 = getelementptr inbounds i16, ptr @masktab, i64 %2
  %4 = load i16, ptr %3, align 2, !tbaa !6
  %5 = getelementptr inbounds i16, ptr @psd, i64 %2
  %6 = load i16, ptr %5, align 2, !tbaa !6
  %7 = sext i16 %4 to i64
  %8 = getelementptr inbounds i16, ptr @bndpsd, i64 %7
  store i16 %6, ptr %8, align 2, !tbaa !6
  %9 = icmp slt i16 %0, 3
  br i1 %9, label %10, label %83

10:                                               ; preds = %1
  %11 = sext i16 %0 to i32
  %12 = add nsw i32 %11, 1
  %13 = sub nsw i32 2, %11
  %14 = zext i32 %13 to i64
  %15 = add nuw nsw i64 %14, 1
  %16 = icmp ult i32 %13, 3
  br i1 %16, label %67, label %17

17:                                               ; preds = %10
  %18 = icmp ult i32 %13, 15
  br i1 %18, label %46, label %19

19:                                               ; preds = %17
  %20 = and i64 %15, 8589934576
  %21 = insertelement <8 x i16> <i16 poison, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, i16 %6, i64 0
  %22 = getelementptr i16, ptr @psd, i64 %2
  br label %23

23:                                               ; preds = %23, %19
  %24 = phi i64 [ 0, %19 ], [ %34, %23 ]
  %25 = phi <8 x i16> [ %21, %19 ], [ %32, %23 ]
  %26 = phi <8 x i16> [ zeroinitializer, %19 ], [ %33, %23 ]
  %27 = getelementptr i16, ptr %22, i64 %24
  %28 = getelementptr i8, ptr %27, i64 2
  %29 = getelementptr i8, ptr %27, i64 18
  %30 = load <8 x i16>, ptr %28, align 2, !tbaa !6
  %31 = load <8 x i16>, ptr %29, align 2, !tbaa !6
  %32 = add <8 x i16> %30, %25
  %33 = add <8 x i16> %31, %26
  %34 = add nuw i64 %24, 16
  %35 = icmp eq i64 %34, %20
  br i1 %35, label %36, label %23, !llvm.loop !10

36:                                               ; preds = %23
  %37 = add <8 x i16> %33, %32
  %38 = tail call i16 @llvm.vector.reduce.add.v8i16(<8 x i16> %37)
  %39 = icmp eq i64 %15, %20
  br i1 %39, label %81, label %40

40:                                               ; preds = %36
  %41 = add nsw i64 %20, %2
  %42 = trunc i64 %20 to i32
  %43 = add i32 %12, %42
  %44 = and i64 %15, 12
  %45 = icmp eq i64 %44, 0
  br i1 %45, label %67, label %46

46:                                               ; preds = %40, %17
  %47 = phi i64 [ %20, %40 ], [ 0, %17 ]
  %48 = phi i16 [ %38, %40 ], [ %6, %17 ]
  %49 = and i64 %15, 8589934588
  %50 = add nsw i64 %49, %2
  %51 = trunc i64 %49 to i32
  %52 = add i32 %12, %51
  %53 = insertelement <4 x i16> <i16 poison, i16 0, i16 0, i16 0>, i16 %48, i64 0
  %54 = getelementptr i16, ptr @psd, i64 %2
  br label %55

55:                                               ; preds = %55, %46
  %56 = phi i64 [ %47, %46 ], [ %62, %55 ]
  %57 = phi <4 x i16> [ %53, %46 ], [ %61, %55 ]
  %58 = getelementptr i16, ptr %54, i64 %56
  %59 = getelementptr i8, ptr %58, i64 2
  %60 = load <4 x i16>, ptr %59, align 2, !tbaa !6
  %61 = add <4 x i16> %60, %57
  %62 = add nuw i64 %56, 4
  %63 = icmp eq i64 %62, %49
  br i1 %63, label %64, label %55, !llvm.loop !14

64:                                               ; preds = %55
  %65 = tail call i16 @llvm.vector.reduce.add.v4i16(<4 x i16> %61)
  %66 = icmp eq i64 %15, %49
  br i1 %66, label %81, label %67

67:                                               ; preds = %40, %64, %10
  %68 = phi i64 [ %2, %10 ], [ %41, %40 ], [ %50, %64 ]
  %69 = phi i16 [ %6, %10 ], [ %38, %40 ], [ %65, %64 ]
  %70 = phi i32 [ %12, %10 ], [ %43, %40 ], [ %52, %64 ]
  br label %71

71:                                               ; preds = %67, %71
  %72 = phi i64 [ %75, %71 ], [ %68, %67 ]
  %73 = phi i16 [ %78, %71 ], [ %69, %67 ]
  %74 = phi i32 [ %79, %71 ], [ %70, %67 ]
  %75 = add nsw i64 %72, 1
  %76 = getelementptr inbounds i16, ptr @psd, i64 %75
  %77 = load i16, ptr %76, align 2, !tbaa !6
  %78 = add i16 %77, %73
  %79 = add nsw i32 %74, 1
  %80 = icmp eq i32 %79, 4
  br i1 %80, label %81, label %71, !llvm.loop !15

81:                                               ; preds = %71, %64, %36
  %82 = phi i16 [ %38, %36 ], [ %65, %64 ], [ %78, %71 ]
  store i16 %82, ptr %8, align 2, !tbaa !6
  br label %83

83:                                               ; preds = %81, %1
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable
define dso_local i16 @logadd(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #1 {
  %3 = load i16, ptr %0, align 2, !tbaa !6
  %4 = load i16, ptr %1, align 2, !tbaa !6
  %5 = add i16 %4, %3
  ret i16 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = load i16, ptr @masktab, align 4, !tbaa !6
  %2 = sext i16 %1 to i64
  %3 = getelementptr inbounds i16, ptr @bndpsd, i64 %2
  %4 = load <4 x i16>, ptr @psd, align 8, !tbaa !6
  %5 = tail call i16 @llvm.vector.reduce.add.v4i16(<4 x i16> %4)
  store i16 %5, ptr %3, align 2, !tbaa !6
  %6 = load i16, ptr getelementptr inbounds nuw (i8, ptr @bndpsd, i64 2), align 2, !tbaa !6
  %7 = icmp eq i16 %6, 140
  br i1 %7, label %9, label %8

8:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

9:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i16 @llvm.vector.reduce.add.v8i16(<8 x i16>) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i16 @llvm.vector.reduce.add.v4i16(<4 x i16>) #4

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
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
!7 = !{!"short", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = distinct !{!10, !11, !12, !13}
!11 = !{!"llvm.loop.mustprogress"}
!12 = !{!"llvm.loop.isvectorized", i32 1}
!13 = !{!"llvm.loop.unroll.runtime.disable"}
!14 = distinct !{!14, !11, !12, !13}
!15 = distinct !{!15, !11, !13, !12}
