; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr48571-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr48571-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@c = dso_local local_unnamed_addr global [624 x i32] zeroinitializer, align 4

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @bar() local_unnamed_addr #0 {
  %1 = load i32, ptr @c, align 4
  br label %2

2:                                                ; preds = %0, %2
  %3 = phi i32 [ %1, %0 ], [ %7, %2 ]
  %4 = phi i64 [ 1, %0 ], [ %8, %2 ]
  %5 = shl nuw nsw i64 %4, 2
  %6 = getelementptr i8, ptr @c, i64 %5
  %7 = shl i32 %3, 1
  store i32 %7, ptr %6, align 4, !tbaa !6
  %8 = add nuw nsw i64 %4, 1
  %9 = icmp eq i64 %8, 624
  br i1 %9, label %10, label %2, !llvm.loop !10

10:                                               ; preds = %2
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  br label %1

1:                                                ; preds = %1, %0
  %2 = phi i64 [ 0, %0 ], [ %5, %1 ]
  %3 = getelementptr inbounds nuw i32, ptr @c, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 16
  store <4 x i32> splat (i32 1), ptr %3, align 4, !tbaa !6
  store <4 x i32> splat (i32 1), ptr %4, align 4, !tbaa !6
  %5 = add nuw i64 %2, 8
  %6 = icmp eq i64 %5, 624
  br i1 %6, label %7, label %1, !llvm.loop !12

7:                                                ; preds = %1
  tail call void @bar()
  br label %8

8:                                                ; preds = %7, %15
  %9 = phi i64 [ 0, %7 ], [ %17, %15 ]
  %10 = phi i32 [ 1, %7 ], [ %16, %15 ]
  %11 = getelementptr inbounds nuw i32, ptr @c, i64 %9
  %12 = load i32, ptr %11, align 4, !tbaa !6
  %13 = icmp eq i32 %12, %10
  br i1 %13, label %15, label %14

14:                                               ; preds = %8
  tail call void @abort() #3
  unreachable

15:                                               ; preds = %8
  %16 = shl i32 %10, 1
  %17 = add nuw nsw i64 %9, 1
  %18 = icmp eq i64 %17, 624
  br i1 %18, label %19, label %8, !llvm.loop !15

19:                                               ; preds = %15
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

attributes #0 = { nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noreturn nounwind }

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
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11, !13, !14}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
!15 = distinct !{!15, !11}
