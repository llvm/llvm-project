; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/loop-12.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/loop-12.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@p = dso_local local_unnamed_addr global ptr null, align 8
@.str = private unnamed_addr constant [5 x i8] c"abc\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: read, inaccessiblemem: none) uwtable
define dso_local void @foo() local_unnamed_addr #0 {
  %1 = load ptr, ptr @p, align 8, !tbaa !6
  br label %2

2:                                                ; preds = %5, %0
  %3 = phi ptr [ %6, %5 ], [ %1, %0 ]
  %4 = load i8, ptr %3, align 1, !tbaa !11
  switch i8 %4, label %5 [
    i8 10, label %7
    i8 59, label %7
    i8 33, label %7
  ]

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 1
  store ptr %6, ptr @p, align 8, !tbaa !6
  br label %2, !llvm.loop !12

7:                                                ; preds = %2, %2, %2
  ret void
}

; Function Attrs: nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %4, %0
  %2 = phi ptr [ @.str, %0 ], [ %5, %4 ]
  %3 = load i8, ptr %2, align 1, !tbaa !11
  switch i8 %3, label %4 [
    i8 10, label %6
    i8 59, label %6
    i8 33, label %6
  ]

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 1
  br label %1, !llvm.loop !12

6:                                                ; preds = %1, %1, %1
  store ptr %2, ptr @p, align 8, !tbaa !6
  ret i32 0
}

attributes #0 = { nofree norecurse nosync nounwind memory(readwrite, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(write, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!9, !9, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
