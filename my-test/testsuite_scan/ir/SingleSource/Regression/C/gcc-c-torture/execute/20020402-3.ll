; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020402-3.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20020402-3.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable
define dso_local noundef ptr @blockvector_for_pc_sect(i64 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load ptr, ptr %1, align 8, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %5 = load i32, ptr %3, align 8, !tbaa !12
  %6 = icmp sgt i32 %5, 1
  br i1 %6, label %7, label %23

7:                                                ; preds = %2, %7
  %8 = phi i32 [ %21, %7 ], [ %5, %2 ]
  %9 = phi i32 [ %20, %7 ], [ %5, %2 ]
  %10 = phi i32 [ %19, %7 ], [ 0, %2 ]
  %11 = add nuw nsw i32 %8, 1
  %12 = lshr i32 %11, 1
  %13 = add nuw nsw i32 %12, %10
  %14 = zext nneg i32 %13 to i64
  %15 = getelementptr inbounds nuw ptr, ptr %4, i64 %14
  %16 = load ptr, ptr %15, align 8, !tbaa !15
  %17 = load i64, ptr %16, align 8, !tbaa !17
  %18 = icmp ugt i64 %17, %0
  %19 = select i1 %18, i32 %10, i32 %13
  %20 = select i1 %18, i32 %13, i32 %9
  %21 = sub nsw i32 %20, %19
  %22 = icmp sgt i32 %21, 1
  br i1 %22, label %7, label %23, !llvm.loop !21

23:                                               ; preds = %7, %2
  %24 = phi i32 [ 0, %2 ], [ %19, %7 ]
  br label %28

25:                                               ; preds = %28
  %26 = add nsw i32 %29, -1
  %27 = icmp sgt i32 %29, 0
  br i1 %27, label %28, label %36, !llvm.loop !23

28:                                               ; preds = %23, %25
  %29 = phi i32 [ %26, %25 ], [ %24, %23 ]
  %30 = zext nneg i32 %29 to i64
  %31 = getelementptr inbounds nuw ptr, ptr %4, i64 %30
  %32 = load ptr, ptr %31, align 8, !tbaa !15
  %33 = getelementptr inbounds nuw i8, ptr %32, i64 8
  %34 = load i64, ptr %33, align 8, !tbaa !24
  %35 = icmp ugt i64 %34, %0
  br i1 %35, label %36, label %25

36:                                               ; preds = %28, %25
  %37 = phi ptr [ %3, %28 ], [ null, %25 ]
  ret ptr %37
}

; Function Attrs: nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind memory(read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"symtab", !8, i64 0}
!8 = !{!"p1 _ZTS11blockvector", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!13, !14, i64 0}
!13 = !{!"blockvector", !14, i64 0, !10, i64 8}
!14 = !{!"int", !10, i64 0}
!15 = !{!16, !16, i64 0}
!16 = !{!"p1 _ZTS5block", !9, i64 0}
!17 = !{!18, !19, i64 0}
!18 = !{!"block", !19, i64 0, !19, i64 8, !20, i64 16, !16, i64 24, !10, i64 32, !14, i64 36, !10, i64 40}
!19 = !{!"long long", !10, i64 0}
!20 = !{!"p1 _ZTS6symbol", !9, i64 0}
!21 = distinct !{!21, !22}
!22 = !{!"llvm.loop.mustprogress"}
!23 = distinct !{!23, !22}
!24 = !{!18, !19, i64 8}
